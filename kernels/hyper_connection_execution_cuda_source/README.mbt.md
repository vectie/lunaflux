# Hyper-connection envelope lowering

Four AOT stages lower the immutable precision plan: function/control projection
and collapse, positive Sinkhorn, weighted RMSNorm, and residual-stream combine.
Helper and entry-point names are scoped by the caller's prefix, allowing several
envelopes in one module without duplicate device symbols.

The generated arithmetic preserves each declared BF16/F32 round and matrix
orientation. CUDA thread/block choices exist only in the lowering/frame layer,
not in model builders or pure precision plans. Both numerical contracts execute
in the small GB10 oracle fixture; GLM uses the transposed rounded-BF16 law.
