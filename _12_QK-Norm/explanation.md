## QK Normalization (QK-Norm)

QK Normalization (QK-Norm) is a technique used in transformer models to improve the **stability of the attention mechanism** by normalizing the **query (Q)** and **key (K)** vectors before computing attention scores. Instead of relying solely on the standard scaling factor `1 / √d`, QK-Norm ensures that the magnitudes of the query and key vectors remain well-behaved, leading to more stable and reliable attention calculations.

In standard **Scaled Dot-Product Attention**, attention scores are computed as:

```text
Attention Scores = (QKᵀ) / √d
```

where `d` is the dimension of each attention head. The scaling factor prevents the dot products from becoming excessively large as the embedding dimension increases. However, during training, the magnitudes of the query and key vectors themselves can still vary significantly, producing unstable attention distributions.

QK Normalization addresses this issue by first normalizing the query and key vectors, typically using **L2 normalization**, before computing their dot products. The attention calculation becomes:

```text
Q̂ = normalize(Q)

K̂ = normalize(K)

Attention Scores = (Q̂K̂ᵀ) × s
```

where `s` is a learnable scaling parameter or a fixed scaling factor that controls the sharpness of the attention distribution.

By normalizing the vectors, the dot product effectively becomes a measure of **directional similarity** rather than depending heavily on vector magnitude. This makes the attention mechanism less sensitive to changes in feature scale and can improve optimization during training.

You can think of QK Normalization like comparing arrows instead of their lengths. If two arrows point in the same direction, they are considered highly similar regardless of whether one is much longer than the other. By ignoring length and focusing on direction, the comparison becomes more consistent and less affected by arbitrary scaling.

One of the main advantages of QK-Norm is improved numerical stability, especially in large transformer models and long-context settings. It can reduce extremely sharp or unstable attention distributions and help training converge more reliably without significantly increasing computational cost.

Unlike positional encoding methods such as Sinusoidal Embeddings or ALiBi, QK Normalization does not provide information about token positions. Instead, it modifies the internal attention computation itself by normalizing the query and key representations before similarity is measured.

Modern transformer architectures and research models increasingly explore variants of QK Normalization as a simple yet effective way to stabilize attention computations and improve training behavior, particularly in very large-scale language models.
