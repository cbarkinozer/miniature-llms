import math
import torch
import torch.nn as nn
import torch.nn.functional as functional


class SparseTokenSelectionAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        indexer_dimension: int,
        top_k: int,
        maximum_sequence_length: int,
    ):
        super().__init__()

        assert model_dimension % number_of_attention_heads == 0, \
            "model_dimension must be divisible by number_of_attention_heads"

        self.number_of_attention_heads = number_of_attention_heads
        self.attention_head_dimension = (
            model_dimension // number_of_attention_heads
        )
        self.top_k = top_k

        self.query_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.key_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.output_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

        # The indexer is deliberately small: it only needs to score
        # relevance, not carry content, so it is far cheaper than the
        # main attention projections.
        self.indexer_query_projection = nn.Linear(
            model_dimension, indexer_dimension, bias=False
        )
        self.indexer_key_projection = nn.Linear(
            model_dimension, indexer_dimension, bias=False
        )

        causal_mask = torch.tril(
            torch.ones(
                maximum_sequence_length,
                maximum_sequence_length,
                dtype=torch.bool,
            )
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, model_dimension = input_tensor.shape

        causal_mask = self.causal_mask[:sequence_length, :sequence_length]

        # --- Indexer: score every past token's relevance to each query ---
        indexer_query = self.indexer_query_projection(input_tensor)
        indexer_key = self.indexer_key_projection(input_tensor)

        indexer_scores = indexer_query @ indexer_key.transpose(-2, -1)
        indexer_scores = indexer_scores.masked_fill(
            ~causal_mask, float("-inf")
        )

        top_k = min(self.top_k, sequence_length)
        top_k_indices = torch.topk(
            indexer_scores, k=top_k, dim=-1
        ).indices

        selection_mask = torch.zeros_like(causal_mask).unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone()
        selection_mask.scatter_(-1, top_k_indices, True)
        selection_mask = selection_mask & causal_mask.unsqueeze(0)

        # --- Ordinary multi-head attention, restricted to the selection ---
        query = self.query_projection(input_tensor)
        key = self.key_projection(input_tensor)
        value = self.value_projection(input_tensor)

        query = query.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.attention_head_dimension,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.attention_head_dimension,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.attention_head_dimension,
        ).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_dimension
        )

        # Same mask applied to every head.
        attention_scores = attention_scores.masked_fill(
            ~selection_mask.unsqueeze(1), float("-inf")
        )

        attention_weights = functional.softmax(attention_scores, dim=-1)

        attention_output = attention_weights @ value

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, sequence_length, model_dimension
        )

        return self.output_projection(attention_output)
