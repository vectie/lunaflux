# BF16 deployment kernel-root projection

This package turns the opaque complete BF16 release input into an exact,
root-free materialization plan. The deployment root contains exactly one
schema-v2 execution manifest and its content-addressed `.cubin` modules. The
sorted `sha256sum` inventory and producer evidence live outside that payload
root so the existing deployment-bundle assembler continues to enforce its
sole-JSON kernel-root contract.

The package owns no filesystem or process authority. Materialization is a
separate no-overwrite offline operation; runtime code receives only the final
manifest and immutable module root.

An offline caller writes the plan accessors into this exact source shape:

```text
SOURCE/
  kernel-root.plan.v1       # evidence_bytes()
  kernel.files.sha256       # inventory_bytes()
  payload/                  # files(), at each fixed relative_path()
```

Materialize and independently reverify a new deployment input with:

```sh
plan_sha=$(sha256sum "$SOURCE/kernel-root.plan.v1" | awk '{print $1}')
scripts/assemble-luna-kernel-root.sh \
  "$SOURCE#sha256=$plan_sha" "$NEW_OUTPUT"
scripts/verify-luna-kernel-root.sh "$NEW_OUTPUT"
```

The existing release-bundle assembler then receives
`kernel_source_root=$NEW_OUTPUT/kernel-root`,
`kernel_inventory_source=$NEW_OUTPUT/kernel.files.sha256`, and
`kernel_manifest_relative=lunaflux.execution.json`. Neither shell command
compiles, discovers, or substitutes kernel material.
