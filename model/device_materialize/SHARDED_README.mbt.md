# Sharded device-weight materialization

The sharded loader composes the existing approved-file authority with an
immutable dense-Llama tensor-parallel rank plan. Inspection parses and binds
the complete safetensors header and bindings, then deterministically derives
the exact rank plan from the caller's rank, world size, alignment, and arena
ceiling. It validates every scalar source/destination recipe, hashes the
complete file, checks the same pinned handle's stamp, and closes it without
opening a device allocation. Callers never parse the file or supply
source-offset-bearing bindings or plans.

Artifact metadata authenticates the model-content digest and exact weight
bindings; it does not replace the caller's semantic execution identity. Dense
and paged plans for the same authenticated content therefore retain their own
exact plan digest in the derived rank plan. Foreign content is rejected before
rank-plan construction, while the Llama tensor-parallel planner still rejects
a same-content semantic graph whose tensor structure or binding shapes do not
match the authenticated file.

Loading reopens the inspection's private locator and repeats that complete
admission before allocating exactly the local rank arena. A second full-file
pass detects mutations while hashing through one bounded reusable scratch.
Only overlaps with the rank plan's scalar transfer ranges are copied, directly
to final compact device offsets. No complete payload, tensor, or shard host
buffer exists, and a rank never allocates full bytes for a sharded tensor.

Safetensors metadata does not make semantic tensor order authoritative for
physical payload order. Startup therefore retains one fixed integer array of
tensor references sorted by each recipe's first source byte. The transfer
cursor follows that O(tensor-count) order and computes each row/column segment
from scalar recipe strides; it retains no per-segment metadata array.

Source mutation, read failure, or copy failure closes the incomplete local
allocation. A terminal source-close failure also closes an otherwise-ready
allocation before publication. If allocation close fails, the typed cleanup
branch retains the exact rank plan, primary stage/cause, latest cleanup error,
and retry authority. This package owns no scheduler, rank group, NCCL handle,
or topology discovery.
