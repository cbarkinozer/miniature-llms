# Gated DeltaNet (linear attention via the delta rule)

`_19_MambaStateSpace` already showed one alternative to softmax attention: a vector-valued state that gets decayed and written to at every step (a selective state-space model). Gated DeltaNet is a different alternative, built around a different idea — **fast weights**: instead of a vector of "memory", the recurrent state is a small matrix that acts as an associative memory mapping keys to values, and it's updated using the **delta rule** borrowed from classic associative memory models.

**The delta rule, intuitively.** At every step you have a key/value pair `(k_t, v_t)`. Before writing anything, ask the current memory what it *already* predicts for this key: `predicted_value = state @ k_t`. The "delta" is the prediction error: `delta = v_t - predicted_value`. You then write that *error*, not the raw value, into the state: `state = state + outer(k_t, delta)`. If the memory already predicted `v_t` correctly for this key, the delta is near zero and nothing changes — the update is self-correcting, which is why it's called the delta rule (it's the same update rule behind the classic Widrow-Hoff / least-mean-squares learning rule, here applied at inference time as the recurrence itself).

**The gate.** A data-dependent scalar gate (a sigmoid of a learned projection of the current token, same idea as Mamba's selective gate) controls how much of the *old* state survives each step before the new delta is written: `state = gate * state + outer(k_t, delta)`. A gate near 1 keeps old associations; a gate near 0 lets the memory forget quickly and adapt to a new context — exactly the kind of content-dependent forgetting that plain (ungated) linear attention lacks.

**Reading out.** The output at each step is the memory queried by the current query vector, the same way it was queried for the prediction error: `output_t = query_t @ state`.

**Mamba vs. Gated DeltaNet — same goal, different mechanism:**
- Mamba's state is a **vector** updated by a selective decay-and-add rule (`state = decay * state + gate * input`) — it behaves like a gated exponential moving average.
- Gated DeltaNet's state is a **matrix** (a key→value associative memory) updated by the **delta rule** (`state = gate * state + outer(key, prediction_error)`) — it behaves like an online-learned linear map from keys to values, self-correcting rather than just accumulating.

Both replace the O(sequence_length²) cost of softmax attention with an O(sequence_length) recurrence, and both are used in production today: Alibaba's Qwen3.5-4B (mid-2026) is a *dense* model that interleaves three Gated DeltaNet layers with one regular attention layer, repeated throughout the network — using the cheap recurrent mechanism for most of the depth and reserving full attention for periodic global mixing. Moonshot AI's "Kimi Linear" architecture (intended for the not-yet-released Kimi K3) uses the same delta-rule idea under the name KDA (Kimi Delta Attention), again interleaved with periodic full (MLA) attention layers.

This toy implementation runs the recurrence with an explicit Python loop over time steps, exactly like `_19_MambaStateSpace` does — production implementations use a chunked, parallel-friendly formulation of the same recurrence, but the loop here makes the actual update rule easy to read.
