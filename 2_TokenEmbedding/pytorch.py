import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
        Converts token IDS into token embedding vectors.
        Input:  (batch_size, sequence_length)
        Output: (batch_size, seequence_length, embedding_dimension)
    """
    def __init__(self, vocabulary_size:int, embedding_dimension:int) -> None:
        """
            Arguments:
                vocabulary_size : Total number of unique tokens.
                embedding_dimension: Size of each embedding vector.
        """
        super().__init__()
        # learnable lookup table that convert shape to (vocab_size, embedding_dimension)
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension
        )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
            Convert token IDs into embedding vectors.
        """
        return self.embedding(x)