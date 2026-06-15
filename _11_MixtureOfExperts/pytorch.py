import torch
import torch.nn as nn
import torch.nn.functional as F

from _5_SWiGLU.pytorch import MLP


class MOELayer(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        expert_count: int,
        top_k: int = 2,
    ):
        super().__init__()

        self.expert_count = expert_count
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [
                MLP(model_dimension, hidden_dimension)
                for _ in range(expert_count)
            ]
        )

        # Router chooses which experts receive each token
        self.router = nn.Linear(
            model_dimension,
            expert_count,
            bias=False,
        )

    def forward(self, input_tensor):
        """
        input_tensor:
            (batch_size, sequence_length, model_dimension)

        returns:
            (batch_size, sequence_length, model_dimension)
        """

        batch_size, sequence_length, model_dimension = input_tensor.shape

        # Flatten tokens
        flattened_tokens = input_tensor.reshape(
            batch_size * sequence_length,
            model_dimension,
        )

        # Router

        routing_logits = self.router(flattened_tokens)

        routing_probabilities = F.softmax(
            routing_logits,
            dim=-1,
        )

        topk_weights, topk_expert_indices = torch.topk(
            routing_probabilities,
            k=self.top_k,
            dim=-1,
        )

        # Renormalize so selected experts sum to 1
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Final output
        output_tensor = torch.zeros_like(flattened_tokens)

        number_of_tokens = flattened_tokens.shape[0]

        # Process every expert separately

        for expert_index in range(self.expert_count):

            # Shape:
            # (number_of_tokens, top_k)
            expert_assignment_mask = (
                topk_expert_indices == expert_index
            )

            # Skip experts with no assigned tokens
            if not expert_assignment_mask.any():
                continue

            # Which tokens use this expert?
            token_mask = expert_assignment_mask.any(dim=1)

            selected_inputs = flattened_tokens[token_mask]

            # Run expert only once on all its assigned tokens
            selected_outputs = self.experts[expert_index](
                selected_inputs
            )

            # Selected routing weights
            selected_mask = expert_assignment_mask[token_mask]
            selected_weights = topk_weights[token_mask]

            weighted_outputs = torch.zeros_like(selected_outputs)

            # Each token may use this expert in slot 0 or slot 1
            for routing_slot in range(self.top_k):

                slot_mask = selected_mask[:, routing_slot]

                if not slot_mask.any():
                    continue

                weighted_outputs[slot_mask] += (
                    selected_outputs[slot_mask]
                    * selected_weights[slot_mask, routing_slot].unsqueeze(-1)
                )

            output_tensor[token_mask] += weighted_outputs

        return output_tensor.reshape(
            batch_size,
            sequence_length,
            model_dimension,
        )