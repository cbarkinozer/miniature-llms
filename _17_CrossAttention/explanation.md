## Cross-Attention

Cross-Attention is a variation of the attention mechanism used in transformer models where the **queries (Q)** come from one sequence, while the **keys (K)** and **values (V)** come from a different sequence. This allows the model to **connect information between two separate representations**, such as input and output sequences.

In standard **Self-Attention**, a token attends to other tokens within the same sequence. In contrast, Cross-Attention is used when the model needs to integrate information from an external source. This is especially important in encoder-decoder architectures, where the decoder must attend to the encoder’s output.

The core idea is simple:

* Queries represent what the model is trying to find.
* Keys represent available information.
* Values contain the actual information to be retrieved.

In Cross-Attention, queries typically come from the **decoder hidden states**, while keys and values come from the **encoder hidden states**.

The attention computation is the same as in standard scaled dot-product attention:

```text id="cxattn1"
Attention(Q, K, V) = softmax((QKᵀ) / √d) V
```

The difference lies only in where Q, K, and V originate.

You can think of Cross-Attention like a question-and-answer system. The decoder is asking questions (queries), while the encoder provides a database of facts (keys and values). Each query searches through the encoded information and retrieves the most relevant parts to help generate the next output token.

For example, in machine translation, the encoder processes a sentence in the source language, and the decoder generates the translated sentence. At each step, the decoder uses cross-attention to focus on the most relevant words in the source sentence.

A major advantage of Cross-Attention is that it enables **information flow between two different sequences**, making it essential for tasks like translation, summarization, and image captioning. Without cross-attention, the decoder would have no direct access to the encoder’s representations.

Unlike self-attention, where Q, K, and V all come from the same tensor, Cross-Attention separates these roles. This separation allows the model to maintain two distinct representations while still enabling interaction between them.

Modern transformer architectures rely heavily on Cross-Attention in encoder-decoder models, and it remains a foundational mechanism for any task that requires aligning or conditioning one sequence on another.
