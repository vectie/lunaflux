# Persistent device-KV arena

This package owns the single device allocation backing a canonical
`kv/device_layout` value. Semantic preflight validates model identity, layout
version, byte arithmetic, the configured ceiling, and exact first/last regions
before allocation. Construction then performs exactly one allocation and
proves its byte count and physical 256-byte base alignment.

`DeviceKvArena` is the only lifecycle owner. It exposes immutable layout and
checked region metadata, never the allocation or a raw device handle. The
creating worker must keep the exact `device.Context` open until the arena is
closed and must not share the arena across context-owning workers. Region
metadata is not permission to close, copy, or mutate storage.

The arena does not return `device.KernelArgument` values. Those values are
retainable allocation aliases, so documentation alone cannot make them safe
borrows across poison or close. A future paged executor must keep binding and
synchronous launch inside one owner-mediated call. Package-private preparation
can resolve and physically validate the exact component stride beginning at
page zero for a semantic `KvLayerId`; unauthenticated offsets and foreign
region metadata are never accepted.

Any launch failure poisons the arena. Poison is permanent and idempotent:
region validation is rejected afterward, while explicit retry-safe close
authority remains with the arena owner.

Close is explicit and retry-safe. A failed normal close makes the arena
unusable while retaining release authority. If post-allocation construction
validation and cleanup both fail, `CleanupRequired` retains an opaque owner
whose `retry_close` must succeed before the context is closed.

The package allocates no token-step resources. Physical CUDA sanitizer, leak,
soak, and numerical kernel evidence remain required before the Phase 3 gate;
the fake-resource lifecycle tests here establish host-side transition and
ownership behavior only.
