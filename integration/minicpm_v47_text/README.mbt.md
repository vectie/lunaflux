# MiniCPM-V 4.7 text forward

This owner joins the actual flagship decoder with LunaFlux's existing text I/O
plan and prepared frames. It adds token embedding before the 40-layer stack and
selected-row final normalization, full-vocabulary projection and greedy selection
after it. Each prefill/decode phase has 924 ordered launches in the initial
32-token profile. This is the flagship decoder's text path, not a smaller model.

The selected checkpoint has separate embedding and language-head tensors.
Final normalization uses FP32 normalization and one-plus-gamma multiplication
before BF16 publication. The existing head lowering rounds logits to BF16 and
widens them into F32 storage for downstream sampling. Only caller-selected output
rows materialize the vocabulary: the default one-row workspace is 996,672 bytes.

Request tokenization, selected-row/count staging and causal/canvas position
updates remain caller responsibilities. Queue submission allocates no buffers
or launch descriptions. The egress frame owns its preallocated normalization and
logit buffers; other modules, weights, metadata, activations, state and sampled
token buffers are borrowed. Close enclosing executors before this frame, then
release the borrowed resources. Close partial preparation on failure.

`Storage` owns the 22 mutable state/activation/metadata/token allocations, using
152,271,612 bytes in the initial profile. Its budget excludes weights, modules
and the frame's 996,672-byte private output workspace. The ceiling is checked
before allocating. It exposes borrowed typed objects for `Frame::prepare` and
closes partial allocations explicitly. Device allocation does not clear state
or initialize request metadata: the caller supplies valid counts/CSR/page IDs,
positions and reset flags before submission. Kernels perform request-state reset
when those flags are set. Close the frame before storage.

Source/ABI fixtures are software checks, not evidence of CUDA numerical results.
Actual checkpoint loading, device compilation/execution, autoregressive serving,
vision and upstream logit/token comparisons remain separate integration gates.
