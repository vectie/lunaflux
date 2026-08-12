# LunaFlux

> Phase 1 foundation in progress. The repository contains checked MoonBit
> contracts plus bounded configuration, tokenizer JSON, model metadata, and
> safetensors readers; immutable Llama plans and weight bindings; and a private
> CUDA device boundary. It does not yet claim model execution, production CUDA
> readiness, or GPU performance.

LunaFlux is a MoonBit-native, high-throughput language-model inference engine.
Its design combines prefix-aware scheduling, deterministic paged KV memory,
continuous batching, and a small tile-oriented kernel layer without carrying
the Python runtime and compatibility debt of existing inference systems.

**LunaNexa manages the fleet. LunaFlux executes inference.**

~~~mermaid
flowchart LR
    C["OpenAI-compatible or native clients"] --> F["LunaFlux"]
    N["LunaNexa runtime adapter"] --> F
    F --> W["MoonBit scheduler and workers"]
    W --> K["AOT kernel catalog"]
    K --> G["CUDA devices"]
~~~

## Product boundary

LunaFlux owns:

- tokenizer and immutable model-plan construction;
- safetensors loading and device-aware weight placement;
- request admission, continuous batching, chunked prefill, and cancellation;
- fixed-page KV allocation and radix-indexed prefix reuse;
- sampling, streaming events, metrics, and instance health;
- one worker per accelerator and the GPU execution protocol;
- kernel capability selection and the constrained MoonTile kernel IR.

LunaFlux does not own:

- cluster placement, node enrollment, deployment rollout, or tenant governance;
- an artifact registry, container orchestrator, agent runtime, or application UI;
- arbitrary Python model code or untrusted runtime extensions;
- automatic cross-node inference before a topology-specific benchmark gate.

## Design principles

1. MoonBit owns the serving control path.
2. CUDA details stop at one private native ABI.
3. The scheduler is a deterministic single owner of request and KV state.
4. Prefix discovery and physical KV allocation are different concerns.
5. Production kernels are selected from an AOT capability manifest.
6. Unsupported combinations fail during startup, never during a live request.
7. Performance is benchmark evidence, not an architectural claim.
8. Feature breadth follows correctness and steady-state performance.

## Initial release scope

The first useful release targets a single CUDA GPU and one dense,
decoder-only Llama-style architecture:

- BF16 weights loaded from safetensors;
- tokenizer.json BPE tokenization;
- greedy, temperature, top-k, and top-p sampling;
- native streaming protocol plus OpenAI-compatible endpoints;
- continuous batching, paged KV, and chunked prefill;
- prefix caching after the uncached path is proven correct.

Quantization, tensor parallelism, MoE, multimodal models, LoRA, speculative
decoding, and prefill/decode disaggregation are later gated capabilities.

## Repository map

~~~text
contracts/          public protocol and lifecycle vocabulary
cmd/lunaflux/       command-line entry point
config/             validated configuration records
api/                native and compatibility endpoints
tokenizer/          tokenization implementations
model/              model specifications and immutable plans
scheduler/          admission and batch planning
kv/                 page allocator and request block tables
prefix/             radix prefix index
sampling/           token selection
engine/             engine state machine and worker coordination
kernels/            kernel capabilities and MoonTile IR
device/             safe public device abstractions
internal/cuda/      private native bindings and stubs
metrics/            bounded telemetry
benchmarks/         reproducible performance workloads
docs/               contracts, architecture, decisions, and phase gates
~~~

Implemented packages currently cover `contracts/`, focused `config/` records,
the byte-BPE foundation and bounded selected `tokenizer.json` adapter in
`tokenizer/`, validated artifact readers and exact Llama weight bindings in
`model/`, the architecture-neutral operation plan, and the private CUDA/device
discovery boundary. Later packages are created only when their vertical phase
begins; empty architectural packages are deliberately avoided.

## Validation

~~~sh
moon info
moon fmt
moon check --target native --deny-warn
moon test --target native --deny-warn
scripts/validate-boundaries.sh
scripts/validate-cuda-abi.sh
moon run --target native cmd/lunaflux
~~~

## Documents

- [Product contract](docs/PRODUCT_CONTRACT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Detailed implementation plan](docs/PLAN.md)
- [Technical-debt policy](docs/DEBT_POLICY.md)
- [Benchmark contract](docs/BENCHMARKING.md)
- [Architecture decisions](docs/DECISIONS.md)
- [Implementation status](docs/STATUS.md)
