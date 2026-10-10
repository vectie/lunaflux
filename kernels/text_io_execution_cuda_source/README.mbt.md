# Text ingress/egress lowering

The immutable `TextIoPrecision` plan describes BF16 token embeddings, replicated
residual streams, unweighted mean collapse, RMSNorm, selected-row vocabulary
projection and deterministic greedy output. This package emits four AOT kernels;
it does not introduce runtime compilation or model-specific CUDA branching.

`counts[3]` controls live decoder input rows. Separate output-count and selected
row device ports control head work. Output capacity therefore need not grow with
long prefill length. Greedy resolves equal logits by lowest vocabulary index;
invalid selected rows and non-finite logits produce `-1`.

The default mean, normalized activations, weighted activations and projection
output each have explicit BF16 round boundaries; default logits are widened
BF16 in F32. `TextNormalizationLaw` and `TextLogitsLaw` retain these choices in
the precision IR. Learned text egress instead uses F32 normalization through
weight multiplication, one BF16 hidden publication, and unrounded F32 logits.
The serial ordered reductions provide a reference numerical schedule, not
bitwise equivalence to arbitrary framework reduction trees. Non-greedy sampling,
learned residual collapse and lower-precision head weights are separate
semantics and are not silently substituted here. This initial schedule needs
physical numerical/sanitizer validation and throughput optimization before
serving claims.
