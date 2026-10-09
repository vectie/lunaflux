# GLM-5.3 config admission

This package owns bounded, payload-safe admission of the official
`zai-org/GLM-5.3` and `zai-org/GLM-5.3-Flash-BF16` config shapes. It rejects
duplicate keys before map parsing, executable metadata, unknown fields,
profile drift, and mutations of the fixed DSA/KDA, MoE, mHC, MTP, vision, and
control-token schedules.

Successful admission produces only authenticated family metadata. It does not
load files, select kernels, allocate tensors, or grant execution authority.
