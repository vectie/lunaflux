# Gated delta control source

BF16 projections `a/b` and per-head BF16 `A_log/dt_bias` produce FP32 logarithmic
decay and BF16 sigmoid beta. Softplus uses the upstream threshold of 20. Decay
has no artificial lower clamp; values below -5 and negative infinity are valid.

ABI: `counts, a, b, A_log, dt_bias, log_decay, beta`. `counts[3]` is active rows;
projections and outputs are packed `[row, value_head]`. Parameters are shared
per head. The launch is `(maximum_rows, 1, 128)`.

The arithmetic is lowered from typed numeric IR. CUDA compilation and GPU
comparisons remain separate work; this package owns no device resources.
