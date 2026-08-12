# LunaFlux architecture decisions

## ADR-0001 — Independent sibling repository

**Status:** accepted

LunaFlux is an inference execution engine. LunaNexa is a model and cluster
control plane. They have different trust boundaries, release evidence,
dependencies, and failure domains.

LunaFlux therefore lives in its own repository and exposes a provider-neutral
runtime contract. LunaNexa may own an adapter but LunaFlux never imports it.

## ADR-0002 — MoonBit owns the control path

**Status:** accepted

Configuration, tokenization, model planning, scheduling, KV metadata, prefix
reuse, sampling, worker coordination, API contracts, and telemetry are
first-party MoonBit native code.

GPU vendor libraries remain external dependencies behind a private C ABI.
This is MoonBit-native ownership, not a claim that CUDA is reimplemented.

## ADR-0003 — Radix discovery plus fixed-page storage

**Status:** accepted

Token radix trees efficiently discover reusable prefixes. Fixed-size page
arenas make GPU KV capacity deterministic and prevent fragmentation.

The prefix index stores generational page runs and never owns GPU tensors.
The page allocator remains authoritative for physical memory.

## ADR-0004 — AOT production kernels

**Status:** accepted

Production startup selects digest-pinned kernels from a capability manifest.
It does not compile because a request arrived. A developer JIT may be added
later but is disabled in production.

This makes readiness, latency, attack surface, and reproducibility tractable.

## ADR-0005 — Constrained MoonTile before universal DSL

**Status:** accepted

LunaFlux needs tile operations for a finite kernel catalog. It does not need to
rebuild TileLang or TVM before serving one model.

MoonTile grows only from measured kernel requirements. Vendor GEMM remains
valid when it is faster or more reliable.

## ADR-0006 — One dense decoder family first

**Status:** accepted

The initial model plan supports one validated Llama-style dense decoder in
BF16. Breadth follows the immutable plan interface after correctness, paging,
batching, and streaming are proven.

This prevents hundreds of model conditionals from shaping the scheduler.

## ADR-0007 — One scheduler owner

**Status:** accepted

Request lifecycle, page ownership, prefix references, admission, and batch
membership are mutated by one deterministic scheduler owner. API and worker
tasks communicate through bounded typed channels.

This trades uncontrolled shared concurrency for reproducible decisions and
simple invariants while still overlapping CPU planning with GPU execution.

## ADR-0008 — Worker isolation per device

**Status:** accepted

Each accelerator is owned by one worker process. The worker contains CUDA
context and graph state. A worker failure invalidates a device generation
without corrupting scheduler memory.

The initial service has one scheduler process and one worker, not a process per
subsystem.

