# Deterministic deployment-bundle evidence

This package admits the root-free canonical evidence produced by
`scripts/assemble-release-bundle.sh`. The offline assembler takes one
independently digest-pinned explicit input document and a new output path. It
stages separate read-only launch, model, and policy roots beside the exact OCI
context; model bytes are never copied into the image root.

The evidence binds both executables, the authenticated launch/descriptor/policy
documents, every component inventory, the kernel manifest, the rootfs
inventory, the assembly input, and the assembler implementation. Successful
admission is inert. It is suitable for wrapping as a `release/evidence`
artifact, but it does not claim OCI construction, physical CUDA execution, or
production inference.

The shell verifier is authoritative for filesystem exactness. Existing
`deploy/launch_file`, runtime descriptor, instance-policy, execution-manifest,
and OCI validators remain authoritative for their semantic contracts.

`scripts/materialize-release-bundle.sh` adds the atomic semantic workflow. It
claims one new output directory, assembles beneath that private claim, and runs
an independently digest-pinned host `lunaflux validate-materialized-release`
tool before any entry is published. Typed no-follow source-root views bind the
staged bytes to the final absolute labels encoded by the launch and bootstrap
contracts. The completed bundle retains the root-free semantic record, its
digest, the exact tool/materializer digests, and explicit source-target and
no-overwrite flags. Any semantic failure or partial final transfer removes only
the still-authenticated empty claim and leaves no output. Once the fully
verified private stage is installed, publication is a fixed six-entry prefix
transaction. `scripts/recover-release-materialization.sh` explicitly validates
the canonical output, current uid, mode-400 single-link v2 claim, prepared
record, split inventories, filesystem types, and prefix state before completing
publication. Every ambiguous or substituted state is refusal-only; recovery
contains no recursive output deletion.
