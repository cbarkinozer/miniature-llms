import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTokenPredictionLM(nn.Module):
    """
    Multi-Token Prediction (MTP) head.

    Predicts k future tokens for each position using k separate heads.
    """
    def __init__(
        self,
        model_dimension: int,
        vocab_size: int,
        horizon: int = 3,
        tie_weights: bool = False,
    ):
        super().__init__()

        self.horizon = horizon

        # Shared backbone projection per step (optional design choice)
        self.projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        # One LM head per भविष्य step
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(model_dimension, vocab_size, bias=False)
                for _ in range(horizon)
            ]
        )

        # Optional weight tying for first head
        if tie_weights:
            self.lm_heads[0].weight = self.projection.weight

    def forward(self, hidden_states, targets=None):
        """
        hidden_states:
            (batch_size, sequence_length, model_dimension)

        targets:
            (batch_size, sequence_length)

        returns:
            logits:
                (batch_size, sequence_length, horizon, vocab_size)
        """

        batch_size, seq_len, _ = hidden_states.shape

        # Shared transformation
        x = self.projection(hidden_states)

        # Predict k future tokens
        logits = []

        for i in range(self.horizon):
            step_logits = self.lm_heads[i](x)
            logits.append(step_logits)

        logits = torch.stack(
            logits,
            dim=2,  # horizon dimension
        )

        if targets is None:
            return logits

        # Compute multi-step loss
        loss = 0.0

        for i in range(self.horizon):
            # shift targets for future step
            if i + 1 >= seq_len:
                break

            step_logits = logits[:, :- (i + 1), i, :]
            step_targets = targets[:, (i + 1):]

            loss = loss + F.cross_entropy(
                step_logits.reshape(-1, step_logits.size(-1)),
                step_targets.reshape(-1),
            )

        loss = loss / self.horizon

        return logits, loss