# Qwen BF16 offline release binder

This command regenerates the pure candidate set, joins deterministic AOT
compile receipts and produces a startup-bound release. It does not open a
device or compile kernels and never runs in the token-step path.

The optional `--device-target MAJOR MINOR` matches the candidate exporter's
target option. The default remains `12 0` for existing callers. One explicit
target supplies the regenerated recipes, projection tuning architecture,
compiled-set and per-operation receipt checks, bootstrap execution identity,
and printed `target=sm_...` field. Compilation for another target requires a
new export and compiled set; changing an argument cannot relabel an existing
CUBIN. The existing producer remains responsible for supported targets.

Compiler version and digest are separate from device compute capability.
Neither is inferred from a hostname or model family. On Spark use
`--device-target 12 1` at both export and binding. Failures now flush a bounded
diagnostic before aborting, including a compiled-set mismatch rather than an
empty captured output.

Target and option-composition regressions run in `device_target_wbtest.mbt`.
An accepted parser target is not physical qualification or serving promotion.
