# Learned gated pooling lowering

Five prepared effects implement reserve, joint BF16-parameter/F32 projection,
per-feature learned softmax pooling, learned RMSNorm and history publication.
The immutable semantic plan chooses ratio and preceding-window overlap; CUDA
only supplies the physical launch geometry. This is not GLM's mean pooling.

State persists across arbitrary contiguous chunks and singleton decode. The
previous-window first feature bank and current-window second feature bank are
combined only when overlap is requested. Pooled rows are compact, with explicit
counts and group-start positions for subsequent rotation/cache transforms.

The matching prepared frame owns retained state and bounded compact outputs.
GB10 small-fixture numeric/state, memory/leak, race and synchronization checks
pass; checkpoint binding, rotary, quantization and compressed-cache publication
remain required before model use. This is a correctness-first implementation,
not a whole-model accuracy or performance result.
