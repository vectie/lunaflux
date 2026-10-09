# Decoder foundation CUDA source

This family-neutral package renders deterministic BF16 token-embedding and
RMSNorm CUDA source from exact geometry. It deliberately owns no model
identity, compiler invocation, artifact bytes, loader, launch permission, or
qualification state; family adapters bind those authorities separately.

The paired RMSNorm renderer supports two independent BF16 row widths and
weight vectors in one deterministic source kernel. Its counts-first ABI bounds
the live `counts[3]` prefix by a baked maximum row count and uses explicit
ordered F32 multiply, add, and divide operations before BF16-RNE output.
