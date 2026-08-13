# Device-step allocation gate

This native release executable measures the real public
`DeviceStepOwner::stage`/`finish` path after startup preparation and one warm
cycle. Consecutive submitted plans are built before the counter starts, and a
force-included header redirects generated allocation entry points into a
thread-confined counter.

The same header routes the public device calls through a test-only fake CUDA
context. Its fixed-buffer H2D operation performs bounded copies into persistent
fake allocations and the harness checks exactly eight transfers per measured
stage. It does not replace physical-CUDA correctness, sanitizer, or leak gates.

The harness first proves both generated record allocation and fixed-array
allocation are independently visible. A zero result is accepted only after
those positive controls fire.

Run it through:

```sh
scripts/validate-device-step-allocations.sh
```
