import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        head_dimension: int,
        max_sequence_length: int,
        base: float = 10000.0,
        use_yarn: bool = False,
        original_context_length: int = None,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        if head_dimension % 2 != 0:
            raise ValueError("head_dimension must be even for RoPE.")

        self.head_dimension = head_dimension
        self.max_sequence_length = max_sequence_length
        self.use_yarn = use_yarn

        half_dim = head_dimension // 2

        # 1. Standard inverse frequencies
        freq_seq = torch.arange(half_dim, dtype=torch.float32)
        inverse_frequency = base ** (-freq_seq / half_dim)

        # 2. YaRN modification (frequency scaling)
        if use_yarn:
            if original_context_length is None:
                raise ValueError("original_context_length must be set when use_yarn=True")

            # ratio of extension
            ratio = max_sequence_length / original_context_length

            # YaRN-style smoothing factor
            scale = min(1.0, ratio) ** scaling_factor

            # split low/high frequencies
            cutoff = int(half_dim * 0.5)

            inverse_frequency = inverse_frequency.clone()
            inverse_frequency[cutoff:] *= scale

        # 3. Build angles
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # expand to full head dim
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, heads, head_dim]
        """
        _, seq_len, _, head_dim = x.shape

        if head_dim != self.head_dimension:
            raise ValueError(
                f"Expected head dim {self.head_dimension}, got {head_dim}"
            )

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(2)

        return x * cos + rotate_half(x) * sin