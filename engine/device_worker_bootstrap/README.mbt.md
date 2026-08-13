# Device worker bootstrap composition

This internal native startup boundary composes one received startup contract
and decoded canonical bootstrap source into an opaque device-worker owner. It
authenticates the source before acquiring fixed inherited root roles, converts
all locators and digests to typed values, and applies the fixed
`DenseLlamaPagedAotV1` recipe before large graph, weight, module, or device
allocation.

The model configuration and weight inspection use the model root. The paged
execution manifest and AOT modules use the kernel root. The kernel root closes
before device preparation; the model root closes before readiness is queried
and exactly matched. No transport `Ready` frame is written here.

Preparation failures are raised only after every acquired authority closes.
If worker or root cleanup fails, `CleanupRequired` retains each remaining
authority and records independent worker/model/kernel cleanup status for retry.
The ready aggregate exposes only readiness, frame execution, and close; no
root, locator, inspection, plan, module, allocation, or native handle escapes.
