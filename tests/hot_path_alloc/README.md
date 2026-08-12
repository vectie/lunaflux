# Scheduler hot-path allocation gate

This native-release executable instruments the generated MoonBit C for the
single-owner scheduler and reusable worker-protocol path. A force-included
header redirects generated record, array, byte, string, valtype-array, and
external-object allocation calls to thread-confined counters while preserving
their original allocation behavior.

The gate first proves the counter is live with independent record and fixed-
array positive controls. It then constructs and warms the scheduler, both A/B
plan owners, and both A/B completion owners before measuring repeated decode
build, completion write/submit, and retirement cycles. The measured window
crosses a physical-page metadata boundary, so block-table append remains part
of the zero-allocation evidence.

This is runtime instrumentation, not a source scan or disassembly heuristic.
It covers generated MoonBit code and the allocation entry points emitted into
this executable. It does not claim that future subprocess transport, device
execution, CUDA libraries, or the prebuilt MoonBit runtime are allocation-free;
those paths require their own positive-controlled gates when integrated.

Run it with:

```sh
scripts/validate-hot-path-allocations.sh
```
