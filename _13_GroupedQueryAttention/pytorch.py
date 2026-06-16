import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        query_head_count: int,
        key_value_head_count: int,
    ):
        super().__init__()

        assert model_dimension % query_head_count == 0
        assert query_head_count % key_value_head_count == 0, (
            "Query heads must be a multiple of KV heads."
        )

        self.query_head_count = query_head_count
        self.key_value_head_count = key_value_head_count
        self.repetition_count = (
            query_head_count // key_value_head_count
        )
        self.head_dimension = (
            model_dimension // query_head_count
        )

        self.query_projection = nn.Linear(
            model_dimension,
            query_head_count * self.head_dimension,
            bias=False,
        )

        self.key_projection = nn.Linear(
            model_dimension,
            key_value_head_count * self.head_dimension,
            bias=False,
        )

        self.value_projection = nn.Linear(
            model_dimension,
            key_value_head_count * self.head_dimension,
            bias=False,
        )

        self.output_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

    def forward(self, input_tensor, attention_mask=None):
        # input_tensor shape: (batch_size, sequence_length, model_dimension)
        batch_size, sequence_length, model_dimension = input_tensor.shape

        query_states = self.query_projection(input_tensor)
        key_states = self.key_projection(input_tensor)
        value_states = self.value_projection(input_tensor)

        query_states = query_states.view(
            batch_size,
            sequence_length,
            self.query_head_count,
            self.head_dimension,
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size,
            sequence_length,
            self.key_value_head_count,
            self.head_dimension,
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size,
            sequence_length,
            self.key_value_head_count,
            self.head_dimension,
        ).transpose(1, 2)

        # Repeat KV heads so they match the number of query heads
        if self.repetition_count > 1:
            key_states = torch.repeat_interleave(
                key_states,
                repeats=self.repetition_count,
                dim=1,
            )

            value_states = torch.repeat_interleave(
                value_states,
                repeats=self.repetition_count,
                dim=1,
            )

        attention_scores = (
            torch.matmul(
                query_states,
                key_states.transpose(-2, -1),
            )
            / math.sqrt(self.head_dimension)
        )

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                float("-inf"),
            )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_output = torch.matmul(
            attention_weights,
            value_states,
        )

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                model_dimension,
            )
        )

        return self.output_projection(attention_output)