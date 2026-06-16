import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits x into two halves along the last dimension and rotates them.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        head_dimension: int,
        max_sequence_length: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dimension % 2 != 0:
            raise ValueError("head_dimension must be even for RoPE.")

        half_dimension = head_dimension // 2

        frequency_sequence = torch.arange(
            half_dimension, dtype=torch.float32
        )
        inverse_frequency = base ** (
            -frequency_sequence / half_dimension
        )

        positions = torch.arange(
            max_sequence_length, dtype=torch.float32
        )

        angles = torch.outer(positions, inverse_frequency)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Expand from [seq_len, half_dim] -> [seq_len, head_dim]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, sequence_length, num_heads, head_dimension]
        _, sequence_length, _, head_dimension = x.shape

        if head_dimension != self.cos.shape[-1]:
            raise ValueError(
                f"Expected head dimension {self.cos.shape[-1]}, "
                f"got {head_dimension}"
            )

        cos = self.cos[:sequence_length]
        sin = self.sin[:sequence_length]

        # [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        return x * cos + rotate_half(x) * sin