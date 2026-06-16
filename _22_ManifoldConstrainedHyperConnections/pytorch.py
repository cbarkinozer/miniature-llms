import torch
import torch.nn as nn
import torch.nn.functional as functional


def sinkhorn_knopp(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """
    Turns a square matrix of raw logits into an (approximately) doubly
    stochastic matrix: every row and every column sums to 1. This is the
    constraint that keeps the hyper-connection mixing matrix on the
    Birkhoff polytope, so it can only redistribute signal between streams
    instead of amplifying it.
    """
    matrix = torch.exp(logits)

    for _ in range(iterations):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True)

    return matrix


class ManifoldConstrainedHyperConnections(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        stream_count: int,
        sublayer: nn.Module,
        sinkhorn_iterations: int = 5,
    ):
        super().__init__()

        self.stream_count = stream_count
        self.sublayer = sublayer
        self.sinkhorn_iterations = sinkhorn_iterations

        self.reduce_logits = nn.Parameter(torch.zeros(stream_count))
        self.expand_logits = nn.Parameter(torch.zeros(stream_count))
        self.mixing_logits = nn.Parameter(
            torch.eye(stream_count) * 5.0
        )

    @staticmethod
    def expand_to_streams(
        input_tensor: torch.Tensor, stream_count: int
    ) -> torch.Tensor:
        # input_tensor: (batch_size, sequence_length, model_dimension)
        return input_tensor.unsqueeze(2).repeat(1, 1, stream_count, 1)

    def collapse_streams(self, streams: torch.Tensor) -> torch.Tensor:
        reduce_weights = functional.softmax(self.reduce_logits, dim=-1)
        return torch.einsum("s,bnsd->bnd", reduce_weights, streams)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        # streams: (batch_size, sequence_length, stream_count, model_dimension)
        reduce_weights = functional.softmax(self.reduce_logits, dim=-1)
        expand_weights = functional.softmax(self.expand_logits, dim=-1)

        sublayer_input = torch.einsum(
            "s,bnsd->bnd", reduce_weights, streams
        )
        sublayer_output = self.sublayer(sublayer_input)

        contribution = torch.einsum(
            "s,bnd->bnsd", expand_weights, sublayer_output
        )

        mixing_matrix = sinkhorn_knopp(
            self.mixing_logits, self.sinkhorn_iterations
        )
        mixed_streams = torch.einsum(
            "ts,bnsd->bntd", mixing_matrix, streams
        )

        return mixed_streams + contribution
