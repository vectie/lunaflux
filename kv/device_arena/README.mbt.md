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

Paged kernels obtain one owner-mediated, safe `device.KernelArgument` for the
whole arena and derive page offsets from authenticated scalar indices. That
argument is a borrow and must not be launched after arena close begins; page
generation remains host-owned validation state rather than a device pointer.
Kernels that genuinely consume one segment may request a binding by
layer/component/page indices; the arena re-derives the region so it never
trusts unauthenticated caller-provided offsets or foreign region metadata.

Close is explicit and retry-safe. A failed normal close makes the arena
unusable while retaining release authority. If post-allocation construction
validation and cleanup both fail, `CleanupRequired` retains an opaque owner
whose `retry_close` must succeed before the context is closed.

The package allocates no token-step resources. Physical CUDA sanitizer, leak,
soak, and numerical kernel evidence remain required before the Phase 3 gate;
the fake-resource lifecycle tests here establish host-side transition and
ownership behavior only.
