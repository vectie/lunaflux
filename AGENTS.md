# LunaFlux implementation guide

LunaFlux is a documentation-first MoonBit-native inference engine. Before
implementation, read README.mbt.md, docs/PRODUCT_CONTRACT.md,
docs/ARCHITECTURE.md, docs/PLAN.md, and docs/DEBT_POLICY.md.

## Product boundary

LunaFlux owns model loading, tokenization, request scheduling, KV memory,
prefix reuse, sampling, device execution, kernel selection, and instance-level
telemetry. It is not a fleet controller, artifact registry, tenant billing
system, agent runtime, or application platform.

LunaNexa may deploy LunaFlux as an opaque, digest-pinned runtime. LunaFlux must
not import LunaNexa, MoonGate, or any other MoonSuite product.

## Language and runtime

- Write first-party control-path and execution-planning code in MoonBit,
  targeting native builds.
- No Python, PyTorch, or TVM dependency is permitted in the production runtime.
- Keep CUDA, cuBLASLt, and NCCL behind private, narrow native ABI packages.
- GPU resources require explicit deterministic release; never rely only on GC.
- Production kernel artifacts are AOT and content-addressed. Runtime JIT is not
  part of the request path.

## Structure and debt controls

- Use moon.mod and focused packages with their own moon.pkg.
- Public concrete types belong to the public package that owns them.
- internal packages must not leak concrete types through public APIs.
- The scheduler must not import HTTP, model-family, or device-backend packages.
- Model-specific branching belongs in model plan builders.
- Hardware-specific branching belongs in device and kernel catalog packages.
- Prefer files below 500 lines. Files above 800 lines require an architecture
  decision explaining why they cannot be divided.
- No global configuration object or global mutable runtime context.
- No steady-state heap allocation in the token-step scheduling path.
- Authenticate deployment, artifact, and device identity once at startup.
  Production token-step execution must not perform cryptography, filesystem
  validation, canonical evidence rendering, diagnostic host/device round trips,
  or qualification-only scans.

## Validation

Use targeted checks while working. At a completed phase boundary run:

~~~sh
moon info
moon fmt
moon check --target native --deny-warn
moon test --target native --deny-warn
~~~

Kernel or native-ABI changes additionally require sanitizer, deterministic
correctness, leak, and benchmark gates defined in docs/PLAN.md.

The normal developer loop is `moon fmt`, warning-denied native check, and
affected-package tests. Expensive physical, sanitizer, soak, and aggregate
release campaigns run only when their boundary changes or at a phase/release
boundary; they must not be prerequisites for unrelated edits.
