import torch
import torch.nn as nn
import torch.nn.functional as F

from _8_SWiGLU.pytorch import SwiGLUMLP as MLP


class SharedExpertMOE(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        expert_count: int,
        top_k: int = 2,
        shared_expert_count: int = 1,
    ):
        super().__init__()

        self.expert_count = expert_count
        self.top_k = top_k
        self.shared_expert_count = shared_expert_count


        # Shared experts (always active)
        self.shared_experts = nn.ModuleList(
            [
                MLP(model_dimension, hidden_dimension)
                for _ in range(shared_expert_count)
            ]
        )

        # Routed experts (sparse MoE path)
        self.experts = nn.ModuleList(
            [
                MLP(model_dimension, hidden_dimension)
                for _ in range(expert_count)
            ]
        )

        # Router selects experts per token
        self.router = nn.Linear(
            model_dimension,
            expert_count,
            bias=False,
        )

    def forward(self, input_tensor):
        """
        input_tensor:
            (batch_size, sequence_length, model_dimension)
        """

        batch_size, seq_len, model_dim = input_tensor.shape

        # Flatten tokens
        tokens = input_tensor.view(
            batch_size * seq_len,
            model_dim,
        )

        # Shared experts (dense path)
        shared_output = 0.0
        for expert in self.shared_experts:
            shared_output = shared_output + expert(tokens)

        shared_output = shared_output / self.shared_expert_count

        # Router logits
        routing_logits = self.router(tokens)

        routing_probs = F.softmax(routing_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            routing_probs,
            k=self.top_k,
            dim=-1,
        )

        # Normalize top-k weights
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Sparse expert computation
        output = torch.zeros_like(tokens)

        for expert_idx in range(self.expert_count):

            # Find tokens assigned to this expert
            expert_mask = topk_indices == expert_idx

            if not expert_mask.any():
                continue

            token_mask = expert_mask.any(dim=1)

            selected_tokens = tokens[token_mask]

            expert_outputs = self.experts[expert_idx](
                selected_tokens
            )

            selected_expert_mask = expert_mask[token_mask]
            selected_weights = topk_weights[token_mask]

            weighted_output = torch.zeros_like(expert_outputs)

            for k in range(self.top_k):

                slot_mask = selected_expert_mask[:, k]

                if not slot_mask.any():
                    continue

                weighted_output[slot_mask] += (
                    expert_outputs[slot_mask]
                    * selected_weights[slot_mask, k].unsqueeze(-1)
                )

            output[token_mask] += weighted_output

        # Combine shared + routed experts
        output = output + shared_output

        return output.view(batch_size, seq_len, model_dim)