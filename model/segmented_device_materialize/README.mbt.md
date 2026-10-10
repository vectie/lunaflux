# Segmented device materialization

This family-neutral startup package plans one deterministic aligned device
arena from bounded segmented host arenas and ordered tensor regions. The plan
binds the model identity, a caller-pinned source-manifest digest, source arena
lengths, every source range, and every final device offset.

The immutable layout is checked when constructed. Direct, borrowed and streamed
upload consume its existing label without repeating canonical serialization or
SHA-256. Labels associate plans; they do not authenticate payload bytes. Live
arena lengths, device ranges, transfer outcomes and cleanup still get checked.

Materialization allocates exactly once, validates every device region, and
synchronously borrows each `FixedArray` range through the public device API.
Host ownership is never transferred. Successful device allocations require an
explicit close. When a validation or copy failure is followed by allocation
close failure, the incomplete allocation remains behind retryable cleanup
authority and is never published as ready weights.

This package grants no kernel, scheduler, KV-cache, API, dtype interpretation,
or model-family authority.
