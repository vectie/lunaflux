# Device-worker canonical one-row allocation evidence

This native release executable prepares one genuine public
`DeviceWorkerOwner` through the approved-root model, weight, artifact, and
executor path before measurement. It prebuilds 132 canonical plans and
validated plan-frame owners plus one reusable completion-frame owner. The
plans form 44 legitimate three-step request groups, each with its own active
allocator-issued page. Group A uses ordinary prefill, greedy final prefill,
then stochastic decode; group B uses ordinary prefill, stochastic final
prefill, then greedy decode. Every decode consumes the immediately preceding
sampled token. One cycle warms the aggregate; the next 131 cycles complete the
22 six-cycle pairs. Each measured cycle authenticates the frame's exact
request/page generations, row, sampling fields, and expected replay token,
acquires a fresh completion writer, and calls the exact public
`DeviceWorkerOwner::execute_frame`. It authenticates the returned completion
against the submitted plan and expected token; the retained executor separately
enforces the monotonic plan sequence. The harness then retires/resets that plan
owner. All pages remain active through measurement and are released afterward.

A force-included release-only header counts MoonBit managed, array, and string
allocation entry points. Record, `FixedArray`, and retained dynamic-string
positive controls independently trip the same active counter. The target
window remains zero while the fake native device observes exactly eight fixed
H2D copies and one launch/synchronization per admitted graph step and cycle.
Ordinary prefill performs no readback; every producing final/decode row performs
exactly one. No native resource may be created or closed in that window, and
every fake context child must be closed afterward.

Outside the measured window, an ordinary-prefill launch fault, greedy-decode
readback fault, and stochastic-decode non-finite-logit fault traverse the real
aggregate execute path from explicit sequence-one frames. Each exact bounded
failure consumes the injected fault, faults the owner, publishes no completion,
returns the aborted writer owner to reuse, and closes all native resources.
Package-private device-worker tests separately prove finish-before-submit
failure ordering and writer reuse.

The fake device demonstrates allocation, authentication, lifecycle order, and
deterministic sampler replay only. Expected stochastic tokens are computed
before measurement with the same production sampler, so this is not an
independent stochastic or CUDA numerical oracle. It is not sanitizer/leak,
soak, benchmark, or production promotion evidence. The production Llama plan
currently admits one row, so mixed-row/full-batch evidence remains open.

Run it through:

```sh
scripts/validate-device-worker-one-row-allocations.sh
```
