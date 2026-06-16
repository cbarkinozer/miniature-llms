"""
    MiniMax-M3-style dense backbone (mid-2026): not MiniMax M3 itself, but the
    same documented recipe -- a Grouped Query Attention backbone with MiniMax
    Sparse Attention (MSA) layered in for long-context layers. MSA is the
    same indexer + top-k mechanism DeepSeek calls DSA, but applied to real,
    uncompressed GQA key/value blocks instead of a compressed MLA latent --
    M3 does not use MLA. Dense SwiGLU FFN throughout (M3 is not an MoE
    model).
    tokens -> embeddings -> stacked blocks (alternating GQA / sparse-selection
    attention, plain residual, SwiGLU FFN) -> final norm -> logits

    Simplification: as in deepseek-v4/glm-5, the indexer + top-k mechanism is
    taught as a standalone module, so "real KV" sparse selection is modeled
    here as its own attention layer type (not literally reading from GQA's
    internal K/V projections) -- the GQA layers and sparse-selection layers
    are interleaved across depth instead.
"""

import torch
import torch.nn as nn

from _2_TokenEmbedding.pytorch import TokenEmbedding
from _7_RMSNorm.pytorch import RMSNorm
from _8_SWiGLU.pytorch import SwiGLUMLP
from _13_GroupedQueryAttention.pytorch import GQAAttention
from _16_SparseTokenSelectionAttention.pytorch import SparseTokenSelectionAttention


class MiniMaxM3Block(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        key_value_head_count: int,
        indexer_dimension: int,
        top_k: int,
        hidden_dimension: int,
        maximum_sequence_length: int,
        use_sparse_attention: bool,
    ):
        super().__init__()

        self.use_sparse_attention = use_sparse_attention

        if use_sparse_attention:
            self.attention = SparseTokenSelectionAttention(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                indexer_dimension=indexer_dimension,
                top_k=top_k,
                maximum_sequence_length=maximum_sequence_length,
            )
        else:
            self.attention = GQAAttention(
                model_dimension=model_dimension,
                query_head_count=number_of_attention_heads,
                key_value_head_count=key_value_head_count,
            )

        self.ffn = SwiGLUMLP(
            model_dim=model_dimension, hidden_dim=hidden_dimension
        )

        self.attention_norm = RMSNorm(model_dimension=model_dimension)
        self.ffn_norm = RMSNorm(model_dimension=model_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized_input = self.attention_norm(input_tensor)

        if self.use_sparse_attention:
            attention_output = self.attention(normalized_input)
        else:
            sequence_length = input_tensor.shape[1]
            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=input_tensor.device,
                )
            ).view(1, 1, sequence_length, sequence_length)
            attention_output = self.attention(
                normalized_input, attention_mask=causal_mask
            )

        hidden_state = input_tensor + attention_output

        normalized_hidden_state = self.ffn_norm(hidden_state)
        return hidden_state + self.ffn(normalized_hidden_state)


class MiniMaxM3(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        number_of_attention_heads: int,
        key_value_head_count: int,
        indexer_dimension: int,
        top_k: int,
        hidden_dimension: int,
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
                MiniMaxM3Block(
                    model_dimension=model_dimension,
                    number_of_attention_heads=number_of_attention_heads,
                    key_value_head_count=key_value_head_count,
                    indexer_dimension=indexer_dimension,
                    top_k=top_k,
                    hidden_dimension=hidden_dimension,
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
