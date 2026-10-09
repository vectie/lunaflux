# CUDA offline compile evidence

This family-neutral startup package admits one exact canonical receipt together
with two byte-identical module snapshots. It recomputes both module SHA-256
digests, requires byte-for-byte determinism, authenticates the caller-expected
receipt digest, and binds source and recipe digests, the complete compiler
policy and toolchain identity, driver identity, CUDA target, function symbol,
and final catalog module digest.

The retained value is deliberately `InertSelfConsistent`. It proves only that
the supplied receipt and bytes agree. It does not invoke a compiler, establish
that the named compiler or driver produced the bytes, authenticate an offline
builder principal, qualify numerical behavior, load a module, or authorize
execution. Producer authority, compiler-execution proof, and execution
authority therefore remain separate typed failures.

The canonical receipt schema is
`lunaflux-cuda-offline-compile-evidence.v1`; fields are newline-delimited in
the exact order implemented by the package. Offline builders emit that record,
while startup callers separately pin its SHA-256 digest.
