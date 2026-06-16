import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        max_sequence_length: int = 5000,
    ):
        super().__init__()

        # Create positional embedding matrix
        positional_embeddings = torch.zeros(
            max_sequence_length,
            model_dimension,
        )

        positions = torch.arange(
            max_sequence_length,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                model_dimension,
                2,
                dtype=torch.float,
            )
            * (-math.log(10000.0) / model_dimension)
        )

        # Even dimensions: sin
        positional_embeddings[:, 0::2] = torch.sin(
            positions * div_term
        )

        # Odd dimensions: cos
        positional_embeddings[:, 1::2] = torch.cos(
            positions * div_term
        )

        # Shape:
        # (1, max_sequence_length, model_dimension)
        positional_embeddings = positional_embeddings.unsqueeze(
            0
        )

        # Register as a non-trainable buffer
        self.register_buffer(
            "positional_embeddings",
            positional_embeddings,
        )

    def forward(self, input_tensor):
        """
        input_tensor shape:
            (batch_size, sequence_length, model_dimension)
        """

        sequence_length = input_tensor.size(1)

        return (
            input_tensor
            + self.positional_embeddings[
                :, :sequence_length
            ]
        )