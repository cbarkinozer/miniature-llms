# ALiBi (Attention with Linear Biases)

ALiBi (Attention with Linear Biases) is a positional encoding technique used in transformer models to provide information about the **relative distance between tokens** without adding explicit positional embeddings to the input. Instead of modifying the token representations, ALiBi directly adjusts the **attention scores** by applying a distance-based linear bias.

In standard transformers, positional information is typically added to the token embeddings before they are processed by the attention mechanism. ALiBi takes a different approach by leaving the token embeddings unchanged and incorporating positional information directly into the attention calculation. This makes the model aware of token positions without requiring separate positional embedding vectors.

The key idea behind ALiBi is simple: tokens that are farther apart receive a larger negative bias in their attention scores. For every attention head, a fixed slope determines how quickly this penalty increases with distance. During attention computation, the bias is added before the softmax operation:

```text
Attention Scores =
(QKᵀ / √d) + Linear Bias
```

where the linear bias is proportional to the distance between token positions.

Different attention heads use different slopes, allowing some heads to focus primarily on nearby tokens while others can more easily attend to distant parts of the sequence. This creates a natural hierarchy of local and global attention patterns without introducing additional trainable parameters.

You can think of ALiBi like a conversation in a large room. People naturally pay more attention to those sitting nearby and slightly less attention to people farther away. Some listeners are highly focused on their immediate neighbors, while others are comfortable listening across the room. The linear bias acts like this distance-based preference, gently encouraging attention toward closer tokens rather than strictly preventing long-range interactions.

One of the main advantages of ALiBi is that it introduces **no learned positional embeddings** and requires only a small bias matrix during attention computation. Because the bias depends only on relative distances, models using ALiBi often generalize well to sequence lengths longer than those seen during training, making it particularly attractive for long-context language models.

Unlike Sliding Window Attention, ALiBi does **not** restrict which tokens can attend to each other. Every token can still attend to every other token in the sequence, but distant tokens receive progressively larger penalties in their attention scores. This preserves full attention while encouraging the model to prefer nearby context when appropriate.

ALiBi has become a popular alternative to traditional positional embeddings because of its simplicity, parameter-free design, and strong ability to extrapolate to longer sequences without architectural changes.
