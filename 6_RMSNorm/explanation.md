# RMSNorm (Root Mean Square Normalization)
RMSNorm is a normalization method used in modern transformer-based large language models.
It improves training stability while reducing computation.

In the original Transformer architecture by Vaswani et al. (2017), Layer Normalization was used. LayerNorm computes the mean and variance of activations. It subtracts the mean to center values. It then divides by the standard deviation. This keeps activations stable during training. It also improves optimization and convergence.

RMSNorm simplifies this process. It removes mean subtraction completely. It does not center the activations. Instead, it normalizes using the root mean square (RMS). The RMS is the square root of the average of squared values. RMSNorm only scales by vector magnitude. It does not shift values around zero.

Modern large language models often prefer RMSNorm. It is cheaper to compute than LayerNorm. It uses fewer operations. It also works well in deep transformer networks. It keeps training stable in practice. Because of this, it is used in models like LLaMA-style architectures.