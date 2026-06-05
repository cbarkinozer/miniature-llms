# SWiGLU
In the original Transformer paper by Vaswani et al. (2017), the MLP (feed-forward) block was very simple:

FFN(x)=W_2(ReLU(W_1x))

So it was just:
Linear → ReLU → Linear

This design works, but it treats every hidden activation independently and has no mechanism to selectively control which features should pass through.

**What changed in modern models?**

Modern LLMs replaced this with gated MLP variants, especially GLU-based layers, because they perform better in practice.

One of the most common is SwiGLU (Swish-Gated Linear Unit).

SwiGLU, instead of one transformation, the input is split into two paths:

* Content path (what to write)
* Gate path (what to allow through)

Formally:

SwiGLU(x) = (x_W1) ⊙ swish(x_W2)
 
Then usually followed by a final linear projection.

Where:

* x_W1= “value / content” x_W2 = “gate”
* swish(x) = x⋅σ(x)
* ⊙ = elementwise multiplication

### Intuition

Instead of treating every neuron equally, the model learns:

What information matters (content branch)
How much of it should pass (gate branch)

So each feature is filtered dynamically.

**Why it replaced ReLU MLPs?**

SwiGLU and similar gated MLPs replaced the original design because:

1. Better expressiveness

Gating allows the model to selectively suppress or amplify features instead of blindly passing them through ReLU.

2. Smoother gradients than ReLU
ReLU can “kill” neurons (zero gradient for negative inputs)
Swish is smooth → better gradient flow

3. Better performance per parameter

Empirically, models with SwiGLU:

* converge faster
* achieve lower loss
* outperform standard FFNs at similar compute

4. Efficient scaling

Even though it looks like “two projections”, it scales well and is now standard in modern LLMs (LLaMA-style models, etc.).

### One-line summary

Old FFN:
Linear → ReLU → Linear (no control over feature flow)

SwiGLU:
Split into content + gate → elementwise filtering → better controlled information flow