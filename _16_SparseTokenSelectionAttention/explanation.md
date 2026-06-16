# Sparse Token-Selection Attention (DSA / MSA)

Sliding Window Attention shrinks attention cost with a **fixed** rule: every token only ever looks at its `window_size` nearest neighbors, decided in advance by the architecture, not by the data. Sparse Token-Selection Attention shrinks cost with a **learned** rule instead: a small side network looks at the current query and the full history, scores every past token for relevance, and only the most relevant ones are actually attended to. The window is dynamic and content-dependent rather than fixed and positional.

**The mechanism, in two parts:**

1. **The indexer.** A tiny projection (far smaller than the main attention head dimension — just enough capacity to score relevance, not to carry content) produces an indexer query and an indexer key for every token. The dot product of the current token's indexer query against every past token's indexer key gives a relevance score per past token. Because the indexer is so small, scoring every past token is cheap relative to running full attention over all of them.
2. **Top-k selection.** For each query token, keep only the `top_k` highest-scoring past tokens (still respecting the causal mask — you can never select a future token). Build a mask that is `True` only at causal AND top-k-selected positions, then run ordinary scaled dot-product attention restricted to that mask. Every other position gets `-inf` before the softmax, exactly like the causal mask is applied elsewhere in this repo.

The result: instead of attending to every one of the `sequence_length` past tokens, each query only attends to `top_k` of them — a learned, per-token sparsity pattern instead of attention's usual quadratic cost, while still letting the model pick *which* tokens matter rather than assuming "nearby" is what matters (as sliding window does).

**Same idea, two different names in production models, mid-2026:**
- DeepSeek calls this **DSA (DeepSeek Sparse Attention)**. They apply the indexer and top-k selection on top of Multi-Head Latent Attention's *compressed latent* — so the thing being selected from is the small `c_kv` latent vector, not a full per-head key.
- MiniMax calls the identical mechanism **MSA (MiniMax Sparse Attention)**, but applies it on top of a plain Grouped Query Attention backbone — the indexer selects from *real, uncompressed* K/V blocks rather than a compressed latent.

In other words: MLA compresses *what* gets cached, sparse token-selection decides *which* of the cached tokens are worth looking at — the two ideas are complementary and can be (and are, by DeepSeek and GLM-5) stacked on top of each other. This module teaches the indexer + top-k mechanism standalone, the same way `_11_KVCache` and `_18_FlashAttention` are taught standalone elsewhere in this repo, since wiring it tightly into MLA or GQA internals at toy scale would obscure the one idea that matters: a cheap side network decides sparsity, instead of a fixed rule.
