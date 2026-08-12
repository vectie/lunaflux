# Persistent device-KV byte layout

`kv/device_layout` derives LunaFlux's canonical persistent-KV arena geometry
from an immutable model-plan `KvCacheGeometry` and narrow startup limits. It is
a pure byte planner: it imports no scheduler, worker protocol, device facade,
kernel implementation, or private native ABI, and it allocates no device or
host arena.

Layout version 1 is layer-major with separately contiguous key and value page
runs:

~~~text
[layer][key-or-value][physical-page][token][kv-head][head-dimension]
~~~

Each physical-page component stores BF16 values and reserves an independently
256-byte-aligned segment; that alignment is owned by layout V1 rather than
selected by callers. The planner checks every multiplication, alignment,
stride, and total arena byte count before returning scalar immutable layout and
region values. Physical page indices are stable offsets; page generations stay
in host ownership metadata and must be authenticated before an executor uses a
region.

This package does not allocate or release a device arena, upload block tables,
write KV values, launch attention kernels, or infer a layout from a model-family
name. Those operations consume this plan in later packages.
