# GLM-5.3 config admission

This package owns bounded, payload-safe admission of the official
`zai-org/GLM-5.3` and `zai-org/GLM-5.3-Flash-BF16` config shapes. It rejects
duplicate keys before map parsing, executable metadata, unknown fields,
profile drift, and mutations of the fixed DSA/KDA, MoE, mHC, MTP, vision, and
control-token schedules.

Flash also accepts the installed ModelOpt NVFP4 storage descriptor: static,
symmetric four-bit floating weights in groups of sixteen, unquantized
activations and KV, with explicit conversion exclusions. It validates an
explicit RMS epsilon against the model semantics. Packed tensor/scale binding
still checks the actual checkpoint representation independently; config
acceptance does not turn excluded BF16 tensors into FP4 weights.

Successful admission produces only authenticated family metadata. It does not
load files, select kernels, allocate tensors, or grant execution authority.
