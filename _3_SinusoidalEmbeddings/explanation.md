## Sinusoidal Positional Embeddings

Sinusoidal Positional Embeddings are a technique used in transformer models to provide information about the **position of tokens in a sequence**. Since the self-attention mechanism processes all tokens in parallel and has no built-in notion of order, positional embeddings allow the model to distinguish between tokens appearing at different locations.

In a transformer, each input token is first converted into a vector representation called a **token embedding**. However, token embeddings alone do not tell the model whether a word appears at the beginning, middle, or end of a sentence. Sinusoidal positional embeddings solve this problem by adding a position-dependent vector to every token embedding before it is processed by the transformer.

Unlike learned positional embeddings, sinusoidal embeddings are generated using fixed mathematical functions based on **sine** and **cosine** waves of different frequencies. For a position `pos` and embedding dimension `i`, the values are computed as:

```
PE(position, 2i)     = sin(position / 10000^(2i / model_dimension))

PE(position, 2i + 1) = cos(position / 10000^(2i / model_dimension))
```

where `model_dimension` is the embedding dimension of the model.

The use of different frequencies allows each position to have a unique representation. Lower dimensions vary slowly across positions, while higher dimensions oscillate more rapidly. Together, these patterns create distinctive positional signatures that the model can use to infer both absolute and relative positions.

You can think of sinusoidal positional embeddings like assigning every seat in a theater a unique combination of flashing lights. One light flashes very slowly, another a little faster, and another extremely quickly. Although no single light uniquely identifies a seat, the combination of all flashing patterns makes every seat distinguishable from every other one.

A major advantage of sinusoidal embeddings is that they contain **no learnable parameters**. Because they are generated from mathematical functions rather than learned during training, they can naturally generalize to sequence lengths longer than those seen during training, provided the model architecture supports them.

Although many modern language models now use alternatives such as learned positional embeddings or Rotary Positional Embeddings (RoPE), sinusoidal positional embeddings remain an important concept because they were the original positional encoding method introduced in the Transformer architecture and clearly illustrate how positional information can be incorporated into self-attention models.