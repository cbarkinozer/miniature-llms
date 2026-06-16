## Multi-Token Prediction (MTP)

Multi-Token Prediction (MTP) is a training approach for transformer-based language models where the model is trained to predict **multiple future tokens at each position**, instead of only predicting the next single token. This extends the standard autoregressive objective by encouraging the model to learn short-term future structure more directly.

In a standard **autoregressive language model**, training is based on next-token prediction. Given a sequence:

```text id="mtp1"
x1, x2, x3, ..., xn
```

the model learns to predict only the next token at each step:

```text id="mtp2"
P(x_{t+1} | x1, ..., xt)
```

However, this forces the model to learn long-range structure indirectly through repeated single-step predictions.

Multi-Token Prediction modifies this objective by training the model to predict a **fixed horizon of future tokens**. Instead of predicting only `x_{t+1}`, the model predicts:

```text id="mtp3"
x_{t+1}, x_{t+2}, ..., x_{t+k}
```

for a chosen prediction horizon `k`.

---

The model typically produces multiple output heads or multiple decoding projections, each responsible for a different future step. The loss is computed across all predicted steps and combined.

This can be written conceptually as:

```text id="mtp4"
Loss = Σ (CE(Head_i(x_t), x_{t+i}))
```

where each head `i` predicts the token `i` steps into the future.

---

You can think of Multi-Token Prediction like planning ahead while speaking. Instead of saying one word at a time without knowing where the sentence is going, the model tries to anticipate a short continuation of the sentence before producing output. This leads to more structured and forward-aware representations.

For example:

Input:

> "The cat sat on the"

Next-token prediction:

> predict only "mat"

Multi-token prediction:

> predict "mat", "and", "looked"

Each position teaches the model not only what comes next, but also what is likely to follow shortly after.

---

One of the main advantages of Multi-Token Prediction is improved **training efficiency**, since each forward pass provides supervision for multiple future tokens. This increases the amount of learning signal per sequence and can help models learn faster.

Another benefit is that it encourages better **short-range planning**, since the model must represent relationships across several upcoming tokens simultaneously rather than only focusing on immediate transitions.

However, Multi-Token Prediction introduces challenges. Because inference is still typically performed one token at a time, there can be a **mismatch between training and generation behavior**. Predicting further future tokens is inherently harder and may introduce noise if not balanced properly.

---

In modern transformer research, Multi-Token Prediction is often used as a complementary objective rather than a replacement for next-token prediction. It is combined with standard autoregressive loss to improve stability and performance while preserving compatibility with standard decoding.

Overall, Multi-Token Prediction enhances language models by encouraging them to learn **short-horizon structure and forward consistency**, making them more efficient learners of sequential data.
