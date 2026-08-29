# Device worker bootstrap composition

This internal native startup boundary composes one received startup contract
and decoded canonical bootstrap source into an opaque device-worker owner. It
authenticates the source before acquiring fixed inherited root roles and routes
only the closed BF16 `DenseLlamaPagedAotV2`/`DenseLlamaPagedAotV3` source
recipes or the symmetric-I8 `DenseLlamaI8PagedAotV6` recipe. Cross-decoding and
fallback are forbidden before large graph, weight, module, or device
allocation. The outer digest-authenticated launch schema selects its current
BF16 v5 or I8 v6 runtime route before this child boundary.

The BF16 recipes derive the exact paged-plan batch envelope from
`WorkerProtocolLimits.max_plan_rows`; v3 additionally carries the admitted Luna
approval needed by its exact manifest path. The I8 recipe rebuilds the numeric
model plan, reauthenticates the numeric weight artifact and schema-v4 execution
manifest, and re-observes the device numeric capability. In every route the
batch-specialized identity must equal startup authority before any weight or
device authority opens. Plan-token and plan-page ceilings are checked with
widened row-by-context and row-by-pages-per-sequence products, while physical
page capacity remains a separate hard bound.

The model configuration and BF16 or numeric-weight inspection use the model
root. The matching schema-v2 BF16 or schema-v4 I8 execution manifest and AOT
modules use the kernel root. The kernel root closes before device preparation;
the model root closes before readiness is queried and exactly matched. No
transport `Ready` frame is written here.

Preparation failures are raised only after every acquired authority closes.
If worker or root cleanup fails, `CleanupRequired` retains each remaining
authority and records independent worker/model/kernel cleanup status for retry.
The ready aggregate exposes only readiness, frame execution, and close; no
root, locator, inspection, plan, module, allocation, or native handle escapes.
Host and fake-device composition evidence does not establish approved
full-model physical CUDA readiness, numerical correctness, production-tokenizer
quality, I8 or tensor-parallel promotion, sanitizer/leak balance, soak, or
benchmark performance.
