import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        head_count: int,
    ):
        super().__init__()

        assert model_dimension % head_count == 0

        self.head_count = head_count
        self.head_dimension = model_dimension // head_count

        # Q comes from "query sequence" (e.g., decoder)
        self.query_projection = nn.Linear(
            model_dimension,
            model_dimension,
            bias=False,
        )

        # K, V come from "context sequence" (e.g., encoder)
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

    def forward(
        self,
        query_tensor,
        context_tensor,
        attention_mask=None,
    ):
        """
        query_tensor:   (batch, target_seq_len, model_dim)
        context_tensor: (batch, source_seq_len, model_dim)
        """

        batch_size, target_len, _ = query_tensor.shape
        _, source_len, _ = context_tensor.shape

        # Project Q, K, V
        q = self.query_projection(query_tensor)
        k = self.key_projection(context_tensor)
        v = self.value_projection(context_tensor)


        # Reshape into heads
        q = q.view(
            batch_size,
            target_len,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            source_len,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            source_len,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)


        # Scaled dot-product attention
        attn_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dimension)

  
        # Optional mask (e.g., padding mask on context)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0,
                float("-inf"),
            )

        attn_probs = F.softmax(attn_scores, dim=-1)


        # Weighted sum of values
        out = torch.matmul(attn_probs, v)


        # Merge heads
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, target_len, -1)
        )

        return self.output_projection(out)