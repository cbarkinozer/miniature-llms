"""
    Kimi-K3-style MoE backbone, built around Moonshot's previewed "Kimi
    Linear" architecture (mid-2026): K3 itself is not released yet -- Moonshot
    has only published the Kimi Linear paper and previewed it as the
    foundation for K3 -- so this composes the documented Kimi Linear pattern
    (three Kimi Delta Attention units, modeled here with Gated DeltaNet, per
    one global Multi-Head Latent Attention unit, repeated) on top of K2's
    known foundation (MoE + MLA).
    tokens -> embeddings -> groups of [3x (GatedDeltaNet + FFN), 1x (MLA +
    FFN)], each FFN a Mixture-of-Experts layer -> final norm -> logits
"""

import torch
import torch.nn as nn

from _2_TokenEmbedding.pytorch import TokenEmbedding
from _7_RMSNorm.pytorch import RMSNorm
from _14_MultiHeadLatentAttention.pytorch import MultiHeadLatentAttention
from _20_GatedDeltaNet.pytorch import GatedDeltaNet
from _23_MixtureOfExperts.pytorch import MOELayer


class KimiK3Block(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        latent_dimension: int,
        rope_head_dimension: int,
        hidden_dimension: int,
        expert_count: int,
        maximum_sequence_length: int,
        use_attention: bool,
    ):
        super().__init__()

        self.mixer_norm = RMSNorm(model_dimension=model_dimension)
        if use_attention:
            self.mixer = MultiHeadLatentAttention(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                latent_dimension=latent_dimension,
                rope_head_dimension=rope_head_dimension,
                maximum_sequence_length=maximum_sequence_length,
            )
        else:
            self.mixer = GatedDeltaNet(
                model_dimension=model_dimension,
                number_of_heads=number_of_attention_heads,
            )

        self.moe_norm = RMSNorm(model_dimension=model_dimension)
        self.moe = MOELayer(
            model_dimension=model_dimension,
            hidden_dimension=hidden_dimension,
            expert_count=expert_count,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized_input = self.mixer_norm(input_tensor)
        hidden_state = input_tensor + self.mixer(normalized_input)

        normalized_hidden_state = self.moe_norm(hidden_state)
        return hidden_state + self.moe(normalized_hidden_state)


class KimiK3(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        number_of_attention_heads: int,
        latent_dimension: int,
        rope_head_dimension: int,
        hidden_dimension: int,
        expert_count: int,
        maximum_sequence_length: int,
        group_count: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=model_dimension,
        )

        layers = []
        for _ in range(group_count):
            for _ in range(3):
                layers.append(
                    KimiK3Block(
                        model_dimension=model_dimension,
                        number_of_attention_heads=number_of_attention_heads,
                        latent_dimension=latent_dimension,
                        rope_head_dimension=rope_head_dimension,
                        hidden_dimension=hidden_dimension,
                        expert_count=expert_count,
                        maximum_sequence_length=maximum_sequence_length,
                        use_attention=False,
                    )
                )
            layers.append(
                KimiK3Block(
                    model_dimension=model_dimension,
                    number_of_attention_heads=number_of_attention_heads,
                    latent_dimension=latent_dimension,
                    rope_head_dimension=rope_head_dimension,
                    hidden_dimension=hidden_dimension,
                    expert_count=expert_count,
                    maximum_sequence_length=maximum_sequence_length,
                    use_attention=True,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.final_normalization = RMSNorm(model_dimension=model_dimension)
        self.language_model_head = nn.Linear(
            model_dimension, vocabulary_size, bias=False
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        hidden_state = self.token_embedding(input_tokens)

        for layer in self.layers:
            hidden_state = layer(hidden_state)

        hidden_state = self.final_normalization(hidden_state)
        return self.language_model_head(hidden_state)
