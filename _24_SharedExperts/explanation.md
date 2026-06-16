# Shared Experts (Mixture of Experts with Shared Components)

Shared Experts is a design pattern used in **Mixture of Experts (MoE)** architectures where some expert networks are **shared across all inputs**, while others remain **sparsely activated per token**. This hybrid approach combines the benefits of dense computation (stability and shared knowledge) with sparse routing (efficiency and specialization).

In a standard **Mixture of Experts model**, each token is routed by a gating network to a small subset of expert networks. Only those selected experts process the token, which reduces computation while allowing the model to scale to a large number of parameters.

However, purely sparse expert routing can sometimes lead to instability or loss of shared global knowledge, since different tokens may activate completely different experts. Shared Experts address this by introducing one or more **always-on expert modules** that process every token, regardless of routing decisions.

The final output is typically computed as a combination of:

* **Shared expert output** (dense path, always active)
* **Selected routed expert outputs** (sparse path, conditional on gating scores)

This can be written conceptually as:

```text id="moe1"
Output = SharedExpert(x) + Σ (Gate_i(x) × Expert_i(x))
```

where `SharedExpert(x)` is applied to all tokens, and the sum runs over a small subset of activated experts.

You can think of Shared Experts like a team structure in an organization. There is a central team that handles general knowledge and ensures consistency across all tasks, while specialized consultants are brought in only when needed for specific problems. This ensures both stability and specialization without overloading the system.

One of the main advantages of Shared Experts is improved **knowledge sharing across tokens and tasks**. The shared component acts as a regularizing force, preventing experts from becoming too isolated or overly specialized. This often improves training stability and generalization.

Another benefit is that Shared Experts reduce the burden on routing. Even if the gating mechanism is imperfect, the shared expert ensures that every token still receives a reliable baseline transformation.

In modern large-scale language models, Shared Experts are often used in combination with **sparse MoE layers**, where a subset of parameters is activated per token while a shared pathway maintains consistent global representation. This allows models to scale to billions or trillions of parameters without proportionally increasing computation.

Overall, Shared Experts provide a practical compromise between dense and sparse computation, improving both robustness and efficiency in Mixture-of-Experts transformer architectures.
