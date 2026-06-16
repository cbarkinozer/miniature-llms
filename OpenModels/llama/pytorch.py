"""
    Modern LLaMA/Mistral-style transformer backbone, but not Llama or Mistral itself, because it lacks components such as sliding-window attention and grouped-query attention.
    tokens → embeddings → transformer blocks (with RoPE inside attention) → final normalization → logits
    This model is fully trainable as a standard decoder-only transformer, so you can feed token sequences, compute vocabulary logits, and train it with next-token prediction using cross-entropy on shifted targets, and you should see the loss decrease over time if the data and implementation are correct; during training it runs with use_cache=False to process full sequences with causal masking and learn from all positions in parallel, while during inference it switches to use_cache=True to generate tokens autoregressively by reusing cached keys and values for efficiency, and because it includes token embeddings, stacked causal attention blocks with residual connections, MLP layers, normalization, and a final projection head, it contains all the necessary components for effective language modeling and stable learning.
"""

import torch
import torch.nn as nn

from _2_TokenEmbedding.pytorch import TokenEmbedding
from _21_ResidualBlock.pytorch import TransformerBlock
from _7_RMSNorm.pytorch import RMSNorm


class Llama(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        number_of_attention_heads: int,
        hidden_dimension: int,
        maximum_sequence_length: int,
        layer_count: int
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocabulary_size=vocabulary_size,
            embedding_dimension=model_dimension
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                model_dimension=model_dimension,
                number_of_attention_heads=number_of_attention_heads,
                hidden_dimension=hidden_dimension,
                maximum_sequence_length=maximum_sequence_length
            )
            for _ in range(layer_count)
        ])

        self.final_normalization = RMSNorm(model_dimension=model_dimension)

        self.language_model_head = nn.Linear(
            model_dimension,
            vocabulary_size,
            bias=False
        )

    def reset_cache(self):
        """
        Call this before starting a new autoregressive generation sequence.
        """
        for layer in self.layers:
            if hasattr(layer.attention, "reset_cache"):
                layer.attention.reset_cache()

    def forward(
        self,
        input_tokens: torch.Tensor,
        use_cache: bool = False,
        past_key_values=None
    ):
        """
        Training:
            use_cache=False, past_key_values=None
            → full sequence forward pass

        Inference:
            use_cache=True
            → incremental token decoding with KV cache
        """

        hidden_state = self.token_embedding(input_tokens)

        next_key_values = [] if use_cache else None

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for layer_index, transformer_layer in enumerate(self.layers):

            layer_past_key_value = past_key_values[layer_index]

            if use_cache:
                hidden_state, new_key_value = transformer_layer(
                    hidden_state,
                    use_cache=True,
                    past_key_value=layer_past_key_value
                )
                next_key_values.append(new_key_value)
            else:
                hidden_state = transformer_layer(
                    hidden_state,
                    use_cache=False,
                    past_key_value=None
                )

        hidden_state = self.final_normalization(hidden_state)

        logits = self.language_model_head(hidden_state)

        if use_cache:
            return logits, next_key_values

        return logits