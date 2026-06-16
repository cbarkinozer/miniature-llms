import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        head_count: int,
    ):
        super().__init__()

        assert model_dimension % head_count == 0

        self.head_count = head_count
        self.head_dimension = (
            model_dimension // head_count
        )

        self.query_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        self.key_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        self.value_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        self.output_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

    def forward(
        self,
        input_tensor,
        attention_mask=None,
    ):
        # input_tensor shape:
        # (batch_size, sequence_length, model_dimension)

        (
            batch_size,
            sequence_length,
            model_dimension,
        ) = input_tensor.shape

        query_states = self.query_projection(
            input_tensor
        )

        key_states = self.key_projection(
            input_tensor
        )

        value_states = self.value_projection(
            input_tensor
        )

        query_states = query_states.view(
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size,
            sequence_length,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        attention_scores = (
            torch.matmul(
                query_states,
                key_states.transpose(-2, -1),
            )
            / math.sqrt(self.head_dimension)
        )

        if attention_mask is not None:
            attention_scores = (
                attention_scores.masked_fill(
                    attention_mask == 0,
                    float("-inf"),
                )
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

        return self.output_projection(
            attention_output
        )