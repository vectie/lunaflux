# Device-worker mixed/full-batch allocation evidence

This native release executable prepares one genuine public
`DeviceWorkerOwner` through the approved-root model, weight, schema-v2
execution-manifest, artifact, and executor path. The paged plan and startup
contract bind an exact four-row batch identity. Its worker and device-step
envelopes hold exactly eight query tokens, eight page-table entries, 32
capability entries, and four completion slots.

Before measurement the harness builds 66 validated full-batch plans. Every
plan has three prefill rows followed by one decode row: an ordinary prefill,
two final prefills, and one decode. Their token lengths are 3/3/1/1; every row
owns a distinct two-page CSR segment; completion slots are deliberately
permuted. The flattened token, page, capability, row, and slot tables fill all
configured ceilings exactly. Alternating plans swap greedy and stochastic
sampling across both final-prefill and decode rows.

The fake final-head kernel writes nonuniform BF16 logits into all eight query
rows. Each row has row-shifted candidate logits `[4, 3, 2, 1]` and four lower
tokens. Greedy therefore selects a nonconstant, explicitly nonzero token on
physical rows 6 and 7. A test-local scalar oracle independently insertion-sorts
all eight token IDs, applies temperature softmax, top-k, and top-p, then uses
its own counter mixing and weighted interval selection. Candidate
probabilities differ, and the oracle does not call the production sampler.
Every completion is authenticated against the exact
submitted plan and then checked entry-by-entry and slot-by-slot for request,
generation, kind, processed count, and selected token. Successful entries must
also reject the worker-failure accessor.

One full batch warms the owner. The next 65 full batches execute inside the
allocation probe. A force-included release-only header counts MoonBit managed,
array, and string allocation entry points; record, `FixedArray`, and dynamic
string positive controls prove each counter. The measured window must remain
zero while the fake device observes exactly seven fixed H2D copies (six
payloads and the final real-count publication), one launch and synchronization
per admitted graph operation, and three row readbacks per batch. No native
resource may be created or closed in the window.

Outside measurement, focused hostile evidence rejects a fifth row and the
first token/page/capability beyond each exact maximum without corrupting the
owner. An internally consistent wire frame with an overlong page CSR reaches
the real device preflight, fails exact `PageCount`, aborts its completion
writer, faults and closes the separate worker owner, and leaks no page/native
authority. A startup contract whose bootstrap-source digest differs from the
admitted contract is rejected before device preparation. Existing launch,
readback, and nonfinite-logit injections retain their fail-stop and cleanup
coverage.

The generated-C gate hard-fails missing batch lifecycle helpers and scans each
reachable out-of-line success body for managed allocation. This is software
control-flow, ownership, nonuniform fake-logit, deterministic scalar-oracle,
and warmed allocation evidence. It is not physical CUDA numerical,
sanitizer/leak, soak, benchmark, or independent statistical-distribution
promotion evidence.

Run it through:

```sh
scripts/validate-device-worker-batch-allocations.sh
```

The former one-row script name remains a compatibility wrapper for this gate.
