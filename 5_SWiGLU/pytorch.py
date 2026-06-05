import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()

        # gate + up projections (no bias in modern LLMs)
        self.gate_projection = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up_projection = nn.Linear(model_dim, hidden_dim, bias=False)

        # project back to model dimension
        self.down_projection = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_projection(x)              # no activation
        up = F.silu(self.up_projection(x))          # SiLU activation here

        x = gate * up                               # elementwise gating
        x = self.down_projection(x)                 # back to model dim
        return x