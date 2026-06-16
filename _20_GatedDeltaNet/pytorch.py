import torch
import torch.nn as nn


class GatedDeltaNet(nn.Module):
    def __init__(self, model_dimension: int, number_of_heads: int):
        super().__init__()

        assert model_dimension % number_of_heads == 0, \
            "model_dimension must be divisible by number_of_heads"

        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads

        self.query_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.key_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

        # Data-dependent forget gate, one scalar per head per token.
        self.gate_projection = nn.Linear(
            model_dimension, number_of_heads, bias=True
        )

        self.output_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, model_dimension = input_tensor.shape

        query = self.query_projection(input_tensor).view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        )
        key = self.key_projection(input_tensor).view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        )
        value = self.value_projection(input_tensor).view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        )
        gate = torch.sigmoid(
            self.gate_projection(input_tensor)
        )  # (batch_size, sequence_length, number_of_heads)

        # Fast-weight state: a key -> value associative memory per head.
        state = torch.zeros(
            batch_size,
            self.number_of_heads,
            self.head_dimension,
            self.head_dimension,
            device=input_tensor.device,
        )

        outputs = []

        for time_step in range(sequence_length):
            key_t = key[:, time_step]
            value_t = value[:, time_step]
            query_t = query[:, time_step]
            gate_t = gate[:, time_step].unsqueeze(-1).unsqueeze(-1)

            # Delta rule: write only the prediction error for this key.
            predicted_value = torch.einsum(
                "bhi,bhij->bhj", key_t, state
            )
            prediction_error = value_t - predicted_value

            state = gate_t * state + torch.einsum(
                "bhi,bhj->bhij", key_t, prediction_error
            )

            step_output = torch.einsum(
                "bhi,bhij->bhj", query_t, state
            )
            outputs.append(step_output.unsqueeze(1))

        attention_output = torch.cat(outputs, dim=1)
        attention_output = attention_output.view(
            batch_size, sequence_length, model_dimension
        )

        return self.output_projection(attention_output)
