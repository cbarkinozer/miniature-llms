# Manifold-Constrained Hyper-Connections (mHC)

`_21_ResidualBlock` showed the standard recipe: `output = input + sublayer(input)`, a single residual stream carried through every layer. Hyper-Connections (the idea mHC refines) generalize this to **multiple parallel residual streams**: instead of one path carrying information forward, the network carries `n` streams, and at every layer a learned mapping decides how the streams combine to feed the sublayer, and how the streams mix with each other and with the sublayer's output afterward.

Concretely, at each layer:
1. **Reduce**: the `n` streams are combined (a learned weighted sum) into a single vector that is fed into the sublayer (attention or FFN), exactly as before.
2. **Expand**: the sublayer's output is broadcast back out into a contribution for each of the `n` streams (another learned weighting).
3. **Mix**: the streams themselves are mixed with each other via a learned `n x n` matrix before the sublayer's contribution is added — this is what lets information move *between* streams, not just forward along one of them.

**Why this is unstable without a constraint.** That `n x n` mixing matrix is, in the original (unconstrained) Hyper-Connections formulation, just a free set of learned weights. Composed across dozens of layers, small per-layer amplifications compound multiplicatively — in a 27B-parameter model, researchers measured the resulting signal gain exceeding 3000x, which blows up activations and causes training to diverge.

**The fix: constrain the mixing matrix to the Birkhoff polytope.** A doubly stochastic matrix (every row and every column sums to 1, all entries non-negative) cannot amplify the total signal — it can only redistribute it among the streams, the same way a weighted average can't exceed the range of its inputs. The set of all doubly stochastic matrices is called the Birkhoff polytope. mHC's contribution is forcing the mixing matrix to live on that polytope, computed from raw learned logits via a few iterations of **Sinkhorn-Knopp normalization** (alternately rescaling rows and columns to sum to 1 until the matrix converges to doubly stochastic). This keeps the signal magnitude bounded across arbitrarily many layers while still letting the model learn *which* streams should exchange information with which — exactly the property residual connections already had in the single-stream case, just generalized to many streams.

DeepSeek-V4 (mid-2026) uses mHC in place of plain residual connections throughout its transformer blocks, citing it as what makes the multi-stream idea trainable at scale where earlier unconstrained Hyper-Connections attempts were not.
