import torch
import torch.nn as nn


class MambaSSM(nn.Module):
    def __init__(self, model_dimension, state_dim):
        super().__init__()

        self.state_dim = state_dim

        # Input projections
        self.input_embedding = nn.Linear(model_dimension, state_dim)
        self.gate = nn.Linear(model_dimension, state_dim)

        # State transition and input injection
        self.decay = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.input_proj = nn.Linear(state_dim, state_dim)

        # Output projection
        self.out = nn.Linear(state_dim, model_dimension)

        self.model_dimension = model_dimension

    def forward(self, seq):
        """
        seq: (batch_size, sequence_length, model_dimension)
        """
        batch_size, sequence_length, model_dimension = seq.shape

        state = torch.zeros(batch_size, self.state_dim, device=seq.device)

        outputs = []

        for time_step in range(sequence_length):
            token = seq[:, time_step]

            # Input-dependent gate (Mamba idea: selective update)
            gate = torch.sigmoid(self.gate(token))

            embedded = self.input_embedding(token)                      

            # State update (SSM recurrence)
            state = (
                self.decay * state +
                self.input_proj(embedded) * gate
            )

            step_out = self.out(state)
            outputs.append(step_out.unsqueeze(1))

        return torch.cat(outputs, dim=1)
