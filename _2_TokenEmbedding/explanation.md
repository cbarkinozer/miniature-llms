# Token Embedding
After tokenizing the input text, each token is mapped to a unique integer token ID from the model’s vocabulary.

To convert these discrete token IDs into continuous representations that a neural network can process, we use a token embedding layer. This layer acts as a learnable lookup table that maps each token ID to a dense vector of fixed size (the embedding dimension).

In other words:
Input: token IDs (integers)
Output: embedding vectors (continuous, trainable representations)

These embeddings allow the model to represent semantic relationships between tokens in a high-dimensional space.

In PyTorch, this is implemented efficiently using torch.nn.Embedding, which internally stores a weight matrix of shape (vocab_size,embedding_dim).

Each row corresponds to a token’s learned vector representation.

Example usage:
```py
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dimension)
```
During training, these embedding vectors are updated just like other model parameters via backpropagation.