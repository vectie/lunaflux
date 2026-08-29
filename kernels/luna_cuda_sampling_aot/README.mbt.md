# Sampling-reduction Phase-5 boundary

The architectural verdict is intentionally negative: the current LunaFlux
semantic model graph has no sampling operation. It terminates at one
`LanguageModelHead` producing BF16 vocabulary logits. The device-step owner
copies only each producing row into startup-owned fixed storage and invokes the
canonical host greedy or temperature/top-k/top-p sampler before completion
publication.

`admit_host_sampling_boundary` turns that contract into typed evidence. It
requires the unique final-output producer to be a BF16 language-model head with
the exact vocabulary width and records the exact BF16 row byte count. A plan
whose terminal value is merely a hidden state fails closed.

The package also emits one closed BF16 greedy-reducer source fragment. The
production BF16 exporter co-compiles it into the reusable final LM-head CUBIN,
so it does not fabricate a model operation or an unreferenced manifest module.
The authenticated graph bundle pins those exact module bytes. At startup an
explicit `EmbeddedCudaGreedyV1` bootstrap mode admits the fixed reducer symbol,
shape, and eight-byte result cell against that exact LM-head module; a missing
or mismatched reducer fails without host fallback. The absent/default mode is
host sampling and retains greedy plus temperature/top-k/top-p support.

Device mode allocates its result and host scratch once during executor
preparation. Each plan's greedy compatibility is checked while the mandatory
row preflight is already decoding that row. Execution adds no artifact work,
filesystem access, cryptography, diagnostic readback, or heap allocation; only
the bounded result cells needed to publish token completions cross to host.
