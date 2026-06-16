# Multi-Head Attention

Multi-Head Attention (MHA) is a core mechanism used in transformer models that allows the model to **focus on multiple types of relationships in the input at the same time**. Instead of performing attention only once, the model performs several attention operations in parallel using different attention heads. Each head can learn to capture different patterns, making the model more expressive and effective.

In a standard Multi-Head Attention layer, the input is projected into separate **query (Q)**, **key (K)**, and **value (V)** matrices for every attention head. Each head independently computes attention scores by comparing its queries with its keys, determines which pieces of information are most relevant, and produces its own output representation. These outputs are then combined and projected back into a single representation.

The idea behind multiple heads is that different heads can specialize in different kinds of relationships. For example, one head might focus on nearby words, another might capture long-range dependencies, and another might identify grammatical or semantic relationships. By combining these perspectives, the model builds a richer understanding of the input than a single attention operation could provide.

You can think of it like a team of detectives investigating the same case. Each detective (an attention head) has their own notes, clues, and perspective. One may focus on fingerprints, another on witness statements, and another on security footage. Individually they uncover different pieces of information, but when their findings are combined, they provide a much more complete picture of the case.

The main advantage of Multi-Head Attention is its ability to learn diverse relationships simultaneously, improving the model's capacity to understand language and complex patterns. However, giving every head its own separate queries, keys, and values increases both memory usage and computational cost, especially for large models and long input sequences.

Modern transformer architectures are built around Multi-Head Attention because of its strong performance and flexibility. Many later optimizations, such as Grouped Query Attention (GQA) and Multi-Query Attention (MQA), were developed specifically to reduce the memory and computational overhead of MHA while preserving most of its effectiveness.
