# Multi-Head Latent Attention (MLA)

Multi-Head Latent Attention is the technique DeepSeek introduced (DeepSeek-V2/V3) to shrink the KV cache far more aggressively than Grouped Query Attention does, while keeping most of the expressiveness of full Multi-Head Attention.

Recall the problem GQA solves: the KV cache grows with `num_heads * head_dim` per token, and that memory bound limits how long a context you can serve. GQA shrinks it by having several query heads **share** the same key/value heads — fewer distinct K/V heads means a smaller cache, but every head sharing a group is forced to look at the exact same keys and values.

MLA takes a different approach: instead of sharing whole K/V heads, it compresses the key and value information for *all* heads into a single small **latent vector** per token, `c_kv`, via a low-rank down-projection. Each head then **reconstructs** its own key and value from that shared latent using a per-head up-projection. Because the latent is much smaller than `num_heads * head_dim`, that's the only thing you need to cache at inference — and each head still gets its own reconstructed K/V instead of being forced to literally share one with other heads, so MLA keeps more of full MHA's expressiveness than GQA does at a comparable cache size.

**Why you can't just compress and rotate.** RoPE encodes position by rotating each head's key/query vector by an angle that depends on the token's position. If you tried to cache only the compressed latent and apply RoPE to it directly, the rotation and the per-head up-projection don't commute — rotating the latent before reconstruction doesn't equal reconstructing the head's key and then rotating it. So you can't cleanly cache a RoPE'd latent.

MLA's fix is **decoupled RoPE**: split each head's query/key into two pieces.
- A larger **content piece** that comes from the compressed latent (no rotation applied) — this is what actually gets compressed.
- A small **rope piece** that is projected directly from the input (bypassing the latent) and has RoPE applied to it as usual.

The two pieces are concatenated to form the full query/key vector used in the dot product. Only the small rope piece needs position info; the bulk of the head dimension rides on the cheap shared latent. At inference, the cache stores the latent (shared across heads) plus the small per-head rope piece — still dramatically smaller than caching a full key and value per head.

Queries are not compressed here for clarity (DeepSeek's real implementation also low-rank compresses queries, but that only saves activation memory during training, not KV cache size at inference, so it's skipped in this toy implementation).

In short: GQA shrinks the cache by **sharing** K/V heads outright; MLA shrinks it by **compressing** K/V into a shared latent that every head reconstructs from, paired with a small decoupled rotary slice so positional information survives the compression.
