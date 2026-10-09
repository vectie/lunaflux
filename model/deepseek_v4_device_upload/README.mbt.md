# DeepSeek V4 device upload

This family-owned startup adapter keeps raw checkpoint-weight upload separate
from derived token-hash sidecar upload. Both paths delegate allocation, region
validation, synchronous opaque host borrowing, transfer, and cleanup ownership
to the family-neutral segmented device materializer.

The token-hash path plans exactly three ordered I32 source arenas and device
regions for official layers 0, 1, and 2. Its source digest binds the exact model
identity, source host-manifest digest, ordered layer indices, checked table
digests, and byte counts. Layout validation additionally checks source arena
lengths, region order, zero source offsets, aligned device offsets, and total
materialized/device bytes before any device allocation.

Successful materialization returns the existing explicitly releasable
`SegmentedDevicePreparation`. This package does not merge sidecars into the raw
weight allocation, bind CUDA artifact operands to region offsets, launch a
kernel, or grant execution/qualification authority.
