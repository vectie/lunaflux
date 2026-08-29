# BF16 candidate-set offline exporter

This native-only package publishes one opaque, plan-derived BF16 candidate set
as a canonical compiler-input envelope. It accepts no model root, source text,
symbol, compiler executable, process, CUDA device, runtime, or service object.
Every candidate byte and key already belongs to the authenticated producer
result.

`export_new` requires a canonical absolute output whose parent already exists.
It creates a private deterministic sibling staging directory, writes every file
with `CreateNew`, and renames with replacement disabled. Existing output and
staging paths fail closed. Failure cleanup targets only that exact staging
directory.

The published layout is:

```text
OUTPUT/
  export.v1
  candidate.files.sha256
  candidate-root/
    candidate-set.v1
    recipes/<generated-key>.recipe
    sources/<generated-key>.cu
```

The inventory is independent of `candidate-root`, matching
`scripts/build-luna-bf16-kernel-set.sh`. `export.v1` binds the model identity,
target, profile, compiler identity, candidate declaration, inventory, and exact
operation count while explicitly recording that no compiler, device, or
runtime authority was exercised. Publication is offline preparation, not CUDA
execution or readiness evidence.
