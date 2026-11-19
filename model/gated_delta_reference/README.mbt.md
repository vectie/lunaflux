# Gated delta scalar reference

Prepared FP32 recurrent matrices and channel-local short convolution for the
Qwen3.5-family scalar-decay rule. Grouped query/key heads and unequal key/value
dimensions are explicit. `run_into` reuses managed state and scratch. Reset and
copy/restore define request continuation independently of batch ordering.

This is an offline numerical reference. Callers own BF16 stage boundaries;
transcendentals use host math rounded to FP32. It is not a CUDA fallback or a
bitwise claim about PyTorch/FLA/GPU execution. The model's unbounded non-positive
decay, including negative infinity, is not clipped to Kimi Delta's bounded domain.

The formulas follow Transformers `v5.18.0`'s
[`Qwen3.5` implementation](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
Analytical/state tests are software evidence; target comparisons remain required.
