"""
    DeepSeek-V4-style MoE backbone (mid-2026): not DeepSeek-V4 itself, but the
    same documented recipe -- Multi-Head Latent Attention and Sparse
    Token-Selection Attention (DeepSeek calls this DSA) interleaved across
    layers, Manifold-Constrained Hyper-Connections (mHC) in place of plain
    residual connections, and a Mixture-of-Experts FFN.
    tokens -> embeddings -> hyper-connection streams -> stacked blocks
    (alternating MLA / sparse-selection attention, MoE FFN, both wrapped in
    mHC) -> collapse streams -> final norm -> logits

    Simplification: production DSA selects from MLA's compressed latent
    directly; here the two ideas are taught as separate standalone modules
    (consistent with how this repo teaches KV Cache and Flash Attention
    standalone), so this composition alternates between full MLA layers and
    sparse-selection layers across depth, rather than fusing the indexer
    into MLA's internal latent. This mirrors DeepSeek-V4's own description
    of interleaving different attention types across layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from _2_TokenEmbedding.pytorch import TokenEmbedding
from _7_RMSNorm.pytorch import RMSNorm
from _14_MultiHeadLatentAttention.pytorch import MultiHeadLatentAttention
from _16_SparseTokenSelectionAttention.pytorch import SparseTokenSelectionAttention
from _22_ManifoldConstrainedHyperConnections.pytorch import ManifoldConstrainedHyperConnections
from _23_MixtureOfExperts.pytorch import MOELayer


class PreNormWrapper(nn.Module):
    def __init__(self, norm: nn.Module, sublayer: nn.Module):
        super().__init__()
        self.norm = norm
        self.sublayer = sublayer

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.sublayer(self.norm(input_tensor))


class DeepSeekV4Block(nn.Module):
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
        stream_count: int,
        use_sparse_attention: bool,
    ):
        super().__init__()

        if use_sparse_attention:
            attention_sublayer = SparseTokenSelectionAttention(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                indexer_dimension=indexer_dimension,
                top_k=top_k,
                maximum_sequence_length=maximum_sequence_length,
            )
        else:
            attention_sublayer = MultiHeadLatentAttention(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                latent_dimension=latent_dimension,
                rope_head_dimension=rope_head_dimension,
                maximum_sequence_length=maximum_sequence_length,
            )

        self.attention_hyper_connection = ManifoldConstrainedHyperConnections(
            model_dimension=model_dimension,
            stream_count=stream_count,
            sublayer=PreNormWrapper(
                RMSNorm(model_dimension=model_dimension), attention_sublayer
            ),
        )

        moe_sublayer = MOELayer(
            model_dimension=model_dimension,
            hidden_dimension=hidden_dimension,
            expert_count=expert_count,
        )

        self.ffn_hyper_connection = ManifoldConstrainedHyperConnections(
            model_dimension=model_dimension,
            stream_count=stream_count,
            sublayer=PreNormWrapper(
                RMSNorm(model_dimension=model_dimension), moe_sublayer
            ),
        )

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        streams = self.attention_hyper_connection(streams)
        streams = self.ffn_hyper_connection(streams)
        return streams


class DeepSeekV4(nn.Module):
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
        stream_count: int = 4,
    ):
        super().__init__()

        self.stream_count = stream_count

        self.token_embedding = TokenEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=model_dimension,
        )

        self.layers = nn.ModuleList(
            [
                DeepSeekV4Block(
                    model_dimension=model_dimension,
                    number_of_attention_heads=number_of_attention_heads,
                    latent_dimension=latent_dimension,
                    rope_head_dimension=rope_head_dimension,
                    indexer_dimension=indexer_dimension,
                    top_k=top_k,
                    hidden_dimension=hidden_dimension,
                    expert_count=expert_count,
                    maximum_sequence_length=maximum_sequence_length,
                    stream_count=stream_count,
                    use_sparse_attention=(layer_index % 2 == 1),
                )
                for layer_index in range(layer_count)
            ]
        )

        self.final_normalization = RMSNorm(model_dimension=model_dimension)
        self.language_model_head = nn.Linear(
            model_dimension, vocabulary_size, bias=False
        )
        self.stream_reduce_logits = nn.Parameter(torch.zeros(stream_count))

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        hidden_state = self.token_embedding(input_tokens)

        streams = ManifoldConstrainedHyperConnections.expand_to_streams(
            hidden_state, self.stream_count
        )

        for layer in self.layers:
            streams = layer(streams)

        reduce_weights = functional.softmax(self.stream_reduce_logits, dim=-1)
        hidden_state = torch.einsum("s,bnsd->bnd", reduce_weights, streams)

        hidden_state = self.final_normalization(hidden_state)
        return self.language_model_head(hidden_state)
