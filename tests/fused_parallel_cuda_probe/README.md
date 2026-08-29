# Fused parallel CUDA qualification probe

This native-only executable qualifies the two exact, source-owned Phase-5
fused candidates on an exact `sm_120` device. It exports the candidate set,
binds two independently built CUBIN publications and their canonical receipts,
then compares every physical result with the test-owned scalar referees across
the fixed page-boundary shape matrix.

The package is deliberately qualification-only. It creates no manifest entry,
runtime-serving route, deployment approval, release evidence, or promotion
evidence. The runner accepts CUBINs only; PTX and request-time compilation are
outside this probe.

The `export-production` and `run-production` modes cover the canary-free
production V2 route. They derive the exact production QKV, read-only attention,
and residual/RMSNorm candidates, bind deterministic production QKV and
residual CUBIN receipts, and execute QKV then read-only attention through a
capture-required `OrderedKernelExecutor`. The composite is compared with the
independently generated standalone QKV, RoPE, and full paged-attention CUDA
kernels; residual/RMSNorm remains checked against the test-owned CPU referee.
The source validator pins argument order and ABI identity to the
`engine/device_step` production span preparation used by serving.

This lower fixture intentionally ends at attention and has no logits or
sampling step. The separate `tests/approved_model_spawned_physical`
`device-greedy` and `device-greedy-fused-v2` modes now start from the approved
full decode-step executor, cross the literal spawned-child route, and compare
the embedded fixed result with an independent host full-logits referee. Their
source/static gates pass, but neither mode has run on NVIDIA hardware; no
physical or performance result is claimed.
