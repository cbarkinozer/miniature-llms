"""
    GLM-5-style MoE backbone (mid-2026): not GLM-5 itself, but the same
    documented recipe -- Multi-Head Latent Attention combined with Dynamic
    Sparse Attention (the same indexer + top-k mechanism DeepSeek calls DSA),
    plain residual connections (GLM-5 does not use DeepSeek's mHC), and a
    Mixture-of-Experts FFN.
    tokens -> embeddings -> stacked blocks (alternating MLA / sparse-selection
    attention, plain residual, MoE FFN) -> final norm -> logits

    Simplification: same as deepseek-v4 -- the indexer + top-k mechanism is
    taught as a standalone module rather than fused into MLA's internal
    latent, so this composition alternates between full MLA layers and
    sparse-selection layers across depth instead of literally selecting from
    MLA's compressed latent.
"""

import torch
import torch.nn as nn

from _2_TokenEmbedding.pytorch import TokenEmbedding
from _7_RMSNorm.pytorch import RMSNorm
from _14_MultiHeadLatentAttention.pytorch import MultiHeadLatentAttention
from _16_SparseTokenSelectionAttention.pytorch import SparseTokenSelectionAttention
from _23_MixtureOfExperts.pytorch import MOELayer


class GLM5Block(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        latent_dimension: int,
        rope_head_dimension: int,
        indexer_dimension: int,
        top_k: int,
        hidden_dimension: int,
        expert_count: int,
        maximum_sequence_length: int,
        use_sparse_attention: bool,
    ):
        super().__init__()

        if use_sparse_attention:
            self.attention = SparseTokenSelectionAttention(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                indexer_dimension=indexer_dimension,
                top_k=top_k,
                maximum_sequence_length=maximum_sequence_length,
            )
        else:
            self.attention = MultiHeadLatentAttention(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                latent_dimension=latent_dimension,
                rope_head_dimension=rope_head_dimension,
                maximum_sequence_length=maximum_sequence_length,
            )

        self.moe = MOELayer(
            model_dimension=model_dimension,
            hidden_dimension=hidden_dimension,
            expert_count=expert_count,
        )

        self.attention_norm = RMSNorm(model_dimension=model_dimension)
        self.moe_norm = RMSNorm(model_dimension=model_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized_input = self.attention_norm(input_tensor)
        hidden_state = input_tensor + self.attention(normalized_input)

        normalized_hidden_state = self.moe_norm(hidden_state)
        return hidden_state + self.moe(normalized_hidden_state)


class GLM5(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        number_of_attention_heads: int,
        latent_dimension: int,
        rope_head_dimension: int,
        indexer_dimension: int,
        top_k: int,
        hidden_dimension: int,
        expert_count: int,
        maximum_sequence_length: int,
        layer_count: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=model_dimension,
        )

        self.layers = nn.ModuleList(
            [
                GLM5Block(
                    model_dimension=model_dimension,
                    number_of_attention_heads=number_of_attention_heads,
                    latent_dimension=latent_dimension,
                    rope_head_dimension=rope_head_dimension,
                    indexer_dimension=indexer_dimension,
                    top_k=top_k,
                    hidden_dimension=hidden_dimension,
                    expert_count=expert_count,
                    maximum_sequence_length=maximum_sequence_length,
                    use_sparse_attention=(layer_index % 2 == 1),
                )
                for layer_index in range(layer_count)
            ]
        )

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
