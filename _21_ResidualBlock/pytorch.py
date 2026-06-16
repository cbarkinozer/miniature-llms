import torch
import torch.nn as nn

from _10_CausalMultiheadAttention.pytorch import CausalSelfAttention
from _8_SWiGLU.pytorch import SwiGLUMLP
from _7_RMSNorm.pytorch import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, model_dimension: int, number_of_attention_heads: int, hidden_dimension: int, maximum_sequence_length: int):
        super().__init__()

        # attention + mlp
        self.attention = CausalSelfAttention(
            model_dimension,
            number_of_attention_heads,
            maximum_sequence_length
        )

        self.mlp = SwiGLUMLP(
            model_dim=model_dimension,
            hidden_dim=hidden_dimension
        )

        # normalization layers
        self.attention_norm = RMSNorm(model_dimension=model_dimension)
        self.mlp_norm = RMSNorm(model_dimension=model_dimension)

    def forward(
        self,
        input_tensor: torch.Tensor,
        use_cache: bool = False,
        past_key_value=None
    ):
        # pre-norm attention
        normalized_input = self.attention_norm(input_tensor)

        if use_cache:
            attention_output, new_key_value = self.attention(
                normalized_input,
                use_cache=True,
                past_key_value=past_key_value
            )
        else:
            attention_output = self.attention(normalized_input)
            new_key_value = None

        # residual connection
        hidden_state = input_tensor + attention_output

        # pre-norm mlp
        normalized_hidden_state = self.mlp_norm(hidden_state)
        mlp_output = self.mlp(normalized_hidden_state)

        hidden_state = hidden_state + mlp_output

        if use_cache:
            return hidden_state, new_key_value

        return hidden_state