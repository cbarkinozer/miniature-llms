import torch
import torch.nn as nn
import torch.nn.functional as F


class QKNormAttention(nn.Module):
    """
    QK-Norm Attention:
    - Normalize Query and Key vectors before attention
    - Helps stabilize attention logits
    """
    def __init__(
        self,
        model_dimension: int,
        head_count: int,
        eps: float = 1e-6,
        learnable_scale: bool = True,
    ):
        super().__init__()

        assert model_dimension % head_count == 0

        self.head_count = head_count
        self.head_dimension = model_dimension // head_count
        self.eps = eps

        self.query_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        self.key_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        self.value_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        self.output_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        # optional learnable temperature / scaling
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, input_tensor, attention_mask=None):
        """
        input_tensor: (batch, seq_len, model_dim)
        """

        batch_size, seq_len, _ = input_tensor.shape


        # Linear projections
        q = self.query_projection(input_tensor)
        k = self.key_projection(input_tensor)
        v = self.value_projection(input_tensor)

 
        # Reshape into heads
        q = q.view(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)


        # QK Normalization (L2 norm over head dimension)
        q = F.normalize(q, p=2, dim=-1, eps=self.eps)
        k = F.normalize(k, p=2, dim=-1, eps=self.eps)

        # Scaled dot-product attention
        attn_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        attn_scores = attn_scores * self.scale

        # Optional mask (causal or padding)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0,
                float("-inf"),
            )

        attn_probs = F.softmax(attn_scores, dim=-1)


        # Attention output
        out = torch.matmul(attn_probs, v)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )

        return self.output_projection(out)