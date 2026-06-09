import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from _3_RoPE.pytorch import RoPE


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

        causal_mask = torch.tril(
            torch.ones(
                maximum_sequence_length,
                maximum_sequence_length,
                dtype=torch.bool
            )
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, input_tensor: torch.Tensor):
        batch_size, sequence_length, _ = input_tensor.shape

        query = self.query_projection(input_tensor)
        key = self.key_projection(input_tensor)
        value = self.value_projection(input_tensor)

        query = query.view(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension)
        key = key.view(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension)
        value = value.view(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.rotary_positional_encoding(query)
        key = self.rotary_positional_encoding(key)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_dimension)

        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~causal_mask, float("-inf"))

        attention_weights = functional.softmax(attention_scores, dim=-1)

        attention_output = attention_weights @ value

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, sequence_length, self.embedding_dimension)

        return self.output_projection(attention_output)