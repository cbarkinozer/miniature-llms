import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from _4_RoPE.pytorch import RoPE


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        latent_dimension: int,
        rope_head_dimension: int,
        maximum_sequence_length: int,
    ):
        super().__init__()

        assert model_dimension % number_of_attention_heads == 0, \
            "model_dimension must be divisible by number_of_attention_heads"

        self.model_dimension = model_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.content_head_dimension = (
            model_dimension // number_of_attention_heads
        )
        self.rope_head_dimension = rope_head_dimension
        self.total_head_dimension = (
            self.content_head_dimension + rope_head_dimension
        )

        # Shared low-rank compression of K/V: this is the only thing
        # that needs to be cached per token at inference.
        self.kv_down_projection = nn.Linear(
            model_dimension, latent_dimension, bias=False
        )

        # Per-head reconstruction of content key/value from the latent.
        self.key_up_projection = nn.Linear(
            latent_dimension,
            number_of_attention_heads * self.content_head_dimension,
            bias=False,
        )
        self.value_up_projection = nn.Linear(
            latent_dimension,
            number_of_attention_heads * self.content_head_dimension,
            bias=False,
        )

        # Decoupled rotary slice: projected directly from the input,
        # bypassing the compressed latent, so RoPE can be applied cleanly.
        self.query_rope_projection = nn.Linear(
            model_dimension,
            number_of_attention_heads * rope_head_dimension,
            bias=False,
        )
        self.key_rope_projection = nn.Linear(
            model_dimension,
            number_of_attention_heads * rope_head_dimension,
            bias=False,
        )

        # Query content piece is a plain per-head projection (not
        # compressed): query compression only saves activation memory
        # during training, not KV cache size, so it is skipped here.
        self.query_content_projection = nn.Linear(
            model_dimension,
            number_of_attention_heads * self.content_head_dimension,
            bias=False,
        )

        self.output_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

        self.rotary_positional_encoding = RoPE(
            rope_head_dimension, maximum_sequence_length
        )

        causal_mask = torch.tril(
            torch.ones(
                maximum_sequence_length,
                maximum_sequence_length,
                dtype=torch.bool,
            )
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = input_tensor.shape

        latent_kv = self.kv_down_projection(input_tensor)

        key_content = self.key_up_projection(latent_kv)
        value_content = self.value_up_projection(latent_kv)
        query_content = self.query_content_projection(input_tensor)

        key_content = key_content.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.content_head_dimension,
        ).transpose(1, 2)
        value_content = value_content.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.content_head_dimension,
        ).transpose(1, 2)
        query_content = query_content.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.content_head_dimension,
        ).transpose(1, 2)

        query_rope = self.query_rope_projection(input_tensor).view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.rope_head_dimension,
        ).transpose(1, 2)
        key_rope = self.key_rope_projection(input_tensor).view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.rope_head_dimension,
        ).transpose(1, 2)

        query_rope = self.rotary_positional_encoding(
            query_rope.transpose(1, 2)
        ).transpose(1, 2)
        key_rope = self.rotary_positional_encoding(
            key_rope.transpose(1, 2)
        ).transpose(1, 2)

        query = torch.cat([query_content, query_rope], dim=-1)
        key = torch.cat([key_content, key_rope], dim=-1)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(
            self.total_head_dimension
        )

        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(
            ~causal_mask, float("-inf")
        )

        attention_weights = functional.softmax(attention_scores, dim=-1)

        attention_output = attention_weights @ value_content

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, sequence_length, self.model_dimension
        )

        return self.output_projection(attention_output)
