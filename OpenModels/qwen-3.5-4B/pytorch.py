"""
    Qwen3.5-4B-style dense hybrid backbone (mid-2026): not Qwen3.5-4B itself, but
    the same documented pattern -- 8 groups, each group running 3 Gated DeltaNet
    sublayers followed by 1 regular (GQA) attention sublayer, every sublayer
    followed by its own SwiGLU FFN: 8x(3xDeltaNet->FFN -> 1xAttention->FFN).
    The model is dense (no Mixture of Experts), unlike most of the other
    OpenModels here.
    tokens -> embeddings -> 8 groups of [3x (DeltaNet + FFN), 1x (GQA attention + FFN)] -> final norm -> logits

    Simplification: _13_GroupedQueryAttention does its own internal Q/K/V
    projections and does not expose a hook for injecting RoPE, so the
    attention sublayer here runs without rotary position encoding (the
    Gated DeltaNet sublayers still carry positional information implicitly
    through their sequential recurrence). Production Qwen3.5 applies RoPE
    inside its attention layers.
"""

import torch
import torch.nn as nn

from _2_TokenEmbedding.pytorch import TokenEmbedding
from _7_RMSNorm.pytorch import RMSNorm
from _8_SWiGLU.pytorch import SwiGLUMLP
from _13_GroupedQueryAttention.pytorch import GQAAttention
from _20_GatedDeltaNet.pytorch import GatedDeltaNet


class Qwen35Block(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        key_value_head_count: int,
        hidden_dimension: int,
        use_attention: bool,
    ):
        super().__init__()

        self.use_attention = use_attention

        self.mixer_norm = RMSNorm(model_dimension=model_dimension)
        if use_attention:
            self.mixer = GQAAttention(
                model_dimension=model_dimension,
                query_head_count=number_of_attention_heads,
                key_value_head_count=key_value_head_count,
            )
        else:
            self.mixer = GatedDeltaNet(
                model_dimension=model_dimension,
                number_of_heads=number_of_attention_heads,
            )

        self.ffn_norm = RMSNorm(model_dimension=model_dimension)
        self.ffn = SwiGLUMLP(
            model_dim=model_dimension, hidden_dim=hidden_dimension
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        normalized_input = self.mixer_norm(input_tensor)

        if self.use_attention:
            sequence_length = input_tensor.shape[1]
            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=input_tensor.device,
                )
            ).view(1, 1, sequence_length, sequence_length)
            mixer_output = self.mixer(normalized_input, attention_mask=causal_mask)
        else:
            mixer_output = self.mixer(normalized_input)

        hidden_state = input_tensor + mixer_output

        normalized_hidden_state = self.ffn_norm(hidden_state)
        ffn_output = self.ffn(normalized_hidden_state)

        return hidden_state + ffn_output


class Qwen35(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        number_of_attention_heads: int,
        key_value_head_count: int,
        hidden_dimension: int,
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
                    Qwen35Block(
                        model_dimension=model_dimension,
                        number_of_attention_heads=number_of_attention_heads,
                        key_value_head_count=key_value_head_count,
                        hidden_dimension=hidden_dimension,
                        use_attention=False,
                    )
                )
            layers.append(
                Qwen35Block(
                    model_dimension=model_dimension,
                    number_of_attention_heads=number_of_attention_heads,
                    key_value_head_count=key_value_head_count,
                    hidden_dimension=hidden_dimension,
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
