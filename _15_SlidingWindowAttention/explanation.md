## Sliding Window Attention

Sliding Window Attention (SWA) is a technique used in transformer models to make attention **more efficient for long sequences**. Instead of allowing every token to attend to every other token in the sequence, each token is only allowed to attend to a **fixed-size local window** of nearby tokens. This greatly reduces both memory usage and computation while still preserving useful local context.

In standard **Multi-Head Attention (MHA)**, every token compares itself with every other token in the sequence. For a sequence of length `n`, this requires computing `n × n` attention scores, causing the computational cost and memory usage to grow quadratically as sequences become longer.

Sliding Window Attention addresses this problem by restricting each token's attention to a limited neighborhood. For example, if the window size is 512, each token can only attend to the 512 tokens surrounding it rather than the entire sequence. As a result, the amount of computation grows roughly linearly with sequence length instead of quadratically.

You can think of it like reading a very long book. Instead of constantly flipping back through every previous page whenever you read a new sentence, you only keep a few nearby pages open at once. Most of the information needed to understand the current sentence is usually found in its local context, making this approach much more efficient.

The main advantage of Sliding Window Attention is that it significantly reduces memory consumption and computation for long documents. By limiting attention to nearby tokens, models can process much longer sequences without the enormous cost of full attention.

The primary trade-off is that tokens cannot directly interact with distant parts of the sequence. Information must instead propagate across multiple transformer layers, where each layer gradually passes information from one local window to the next. Despite this limitation, Sliding Window Attention performs well for many natural language tasks because important relationships are often local.

Modern long-context language models frequently combine Sliding Window Attention with other techniques, such as global attention tokens or specialized sparse attention patterns, to efficiently capture both local and long-range dependencies while keeping computational costs manageable.

Sliding Window Attention is an hybrid attention approach.

## Hybrid Attention

Hybrid Attention is a transformer attention design that combines multiple attention mechanisms or patterns within a single model to balance efficiency, long-context reasoning, and representational power. Instead of relying on a single attention strategy (like full self-attention), hybrid approaches mix techniques such as full attention, sliding window attention, global attention, sparse attention, or cross-attention.

The motivation behind Hybrid Attention is that different tokens in a sequence do not require the same level of interaction. Some tokens only need to attend locally, while others require access to global context. By combining multiple attention types, the model can reduce computation while still preserving strong performance.

A common hybrid pattern is to combine:

Local attention (Sliding Window) for nearby context
Global attention for important tokens (e.g., special markers or summary tokens)
Optional full attention layers at deeper stages
Cross-attention in encoder-decoder architectures

The attention computation can be thought of as a mixture of attention masks or pathways, where different heads or layers specialize in different roles.

For example, in a long-document model:

Most tokens attend only to a local window (efficient short-range reasoning)
A small subset of tokens attend globally (information routing)
Occasionally, full attention layers integrate global information

You can think of Hybrid Attention like a communication system in a large organization. Most employees only talk to their immediate team (local attention), managers communicate across departments (global attention), and periodic company-wide meetings synchronize everyone (full attention). Different communication modes coexist to make the system both efficient and well-connected.

The key advantage of Hybrid Attention is that it allows transformer models to scale to very long sequences without the quadratic cost of full attention everywhere. At the same time, it avoids the strict locality limitations of purely sparse methods.

Hybrid Attention is widely used in modern long-context models, where different layers are assigned different attention patterns to optimize both speed and performance. This flexible design makes it possible to tailor attention behavior depending on the role of each layer in the network.

Overall, Hybrid Attention is not a single algorithm but a design philosophy, where multiple attention mechanisms are strategically combined to achieve a balance between efficiency and expressiveness.