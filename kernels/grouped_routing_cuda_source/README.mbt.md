# Grouped routing CUDA source

This package renders a family-neutral serial correctness kernel for grouped
expert selection and selected-weight finalization. Corrected scores are used
only for group and expert choice; uncorrected scores are gathered for weights,
optionally normalized, and scaled. The deterministic lower-id tie policy is a
test oracle, not a claim of PyTorch `topk(sorted=False)` tie-order parity.
