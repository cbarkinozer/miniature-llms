import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from _4_RoPE.pytorch import RoPE


# KV Cache is applied to the attention layer
class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dimension: int, number_of_attention_heads: int, maximum_sequence_length: int):
        super().__init__()

        assert embedding_dimension % number_of_attention_heads == 0, \
            "embedding_dimension must be divisible by number_of_attention_heads"

        self.embedding_dimension = embedding_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.attention_head_dimension = embedding_dimension // number_of_attention_heads

        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        self.rotary_positional_encoding = RoPE(
            self.attention_head_dimension,
            maximum_sequence_length
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(maximum_sequence_length, maximum_sequence_length, dtype=torch.bool)
            ),
            persistent=False
        )

        # KV cache storage (used during inference only)
        self.cached_key_tensor = None
        self.cached_value_tensor = None

    def reset_cache(self):
        """
        Call this at the start of a new sequence generation.
        """
        self.cached_key_tensor = None
        self.cached_value_tensor = None

    def forward(self, input_tensor: torch.Tensor, use_cache: bool = False):
        """
        input_tensor:
            shape (batch_size, sequence_length, embedding_dimension)

        use_cache:
            if True, appends to cached keys/values for autoregressive decoding
        """

        batch_size, sequence_length, _ = input_tensor.shape

        query = self.query_projection(input_tensor)
        key = self.key_projection(input_tensor)
        value = self.value_projection(input_tensor)

        query = query.view(
            batch_size, sequence_length,
            self.number_of_attention_heads,
            self.attention_head_dimension
        )

        key = key.view(
            batch_size, sequence_length,
            self.number_of_attention_heads,
            self.attention_head_dimension
        )

        value = value.view(
            batch_size, sequence_length,
            self.number_of_attention_heads,
            self.attention_head_dimension
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.rotary_positional_encoding(query)
        key = self.rotary_positional_encoding(key)

        # KV CACHE
        if use_cache:
            if self.cached_key_tensor is not None:
                key = torch.cat([self.cached_key_tensor, key], dim=2)
                value = torch.cat([self.cached_value_tensor, value], dim=2)

            self.cached_key_tensor = key
            self.cached_value_tensor = value

            sequence_length = key.shape[2]

        # attention scores
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_dimension)

        # causal mask (only applied for full-sequence mode)
        if not use_cache:
            causal_mask = self.causal_mask[:sequence_length, :sequence_length]
            # ~ tilda is bitwise NOT inversion
            attention_scores = attention_scores.masked_fill(~causal_mask, float("-inf"))

        attention_weights = functional.softmax(attention_scores, dim=-1)

        # @ is matrix multiplication
        attention_output = attention_weights @ value

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dimension
        )

        return self.output_projection(attention_output)