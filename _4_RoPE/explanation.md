# RoPE (Rotary Positional Embeddings)

After computing token embeddings, the model also needs information about token order. Without positional information, the model would treat sequences like a bag of words, where word order is ignored. For example, “dog bites man” and “man bites dog” would be indistinguishable in meaning, which is incorrect.

### Absolute Positional Encoding (APE)

In the original Transformer architecture (Vaswani et al., 2017), Absolute Positional Encoding (APE) was used. It encodes position using fixed or learned vectors that are added to token embeddings.

A common variant uses sinusoidal functions:

positional encoding is computed using sin/cos functions then added directly to token embeddings.

Limitations of APE:

It encodes absolute position, not relative relationships between tokens
It does not generalize well to longer sequence lengths beyond training
It is less effective for capturing relative token interactions, which are crucial in language modeling

Because of these limitations, modern LLMs often avoid APE.

### RoPE (Rotary Positional Embedding)

RoPE(x)= x ⋅ cos(θ) + rotate_half(x) ⋅ sin(θ)

Modern transformer architectures commonly use RoPE, introduced to better encode relative position information.

Unlike APE, RoPE does not add positional vectors to embeddings. Instead, it modifies the query and key vectors inside the attention mechanism.

Key idea:

Each token’s query and key vectors are rotated in a position-dependent way.
The rotation encodes position information directly into the attention computation.

This allows the attention score between tokens to naturally depend on:

their relative distance
their order in the sequence
Where RoPE is applied

Important correction:

RoPE is not applied in the embedding layer.

Instead, it is applied inside the self-attention mechanism, specifically to:

Query vectors (Q)
Key vectors (K)

before the dot-product attention is computed.

**Why RoPE is used in modern LLMs?**

RoPE has several advantages:

Encodes relative positions naturally
Works better with longer contexts
Improves extrapolation beyond training sequence lengths
Integrates smoothly into attention without extra learned parameters