# LunaFlux product contract

## 1. Purpose

LunaFlux turns an approved model artifact and one or more local accelerators
into a bounded streaming inference service. It is an execution engine, not a
fleet control plane. Its initial purpose is low-latency and high-throughput
decoder-only language-model inference on CUDA hardware.

The architecture must remain useful when model families, device generations,
kernel implementations, request mixes, and deployment controllers change.

## 2. Product placement

LunaFlux may be launched directly by an operator or deployed as a digest-pinned
runtime by LunaNexa. Consumers may use the native LunaFlux protocol or an outer
OpenAI-compatible adapter.

LunaFlux has no source dependency on LunaNexa. It never receives LunaNexa node
credentials, placement state, organization records, billing state, application
repositories, or product-specific authority.

## 3. Owned responsibilities

LunaFlux owns:

- model and tokenizer compatibility validation;
- model-plan construction and safetensors materialization;
- device discovery within the process allocation;
- kernel capability validation and selection;
- request tokenization, admission, scheduling, cancellation, and streaming;
- prefill, decode, KV page allocation, prefix reuse, and sampling;
- worker health, request-level accounting, and bounded runtime telemetry.

The deployment environment owns:

- which machine, container, devices, model mount, and network policy are used;
- artifact acquisition, signature verification, and immutable image selection;
- external authentication, tenant policy, quota, billing, and fleet routing;
- process restart, container lifecycle, and long-term telemetry storage.

## 4. Trust boundary

LunaFlux accepts only:

- read-only approved model and tokenizer files;
- validated configuration;
- bounded inference requests;
- local device handles assigned to its process;
- cancellation and drain controls.

The model directory must not contain executable Python or loadable arbitrary
plugins. The v1 loader accepts a declared JSON schema, tokenizer assets, and
safetensors only. Pickle and trust-remote-code behavior are forbidden.

CUDA driver libraries, cuBLASLt, and later NCCL are external system
dependencies behind a private native ABI. Their raw handles are not public
MoonBit contract types.

## 5. Native inference contract

The canonical request carries:

- protocol version and request identifier;
- selected loaded model identity;
- token IDs or normalized text input;
- maximum new tokens and context ceiling;
- sampling parameters;
- stop token IDs and stop strings;
- stream preference and deadline;
- cache-scope identity and cache permission;
- opaque trace correlation without tenant semantics.

Compatibility endpoints translate into the canonical request before outer
admission. At the tokenizer boundary, normalized text becomes an immutable
token buffer. The scheduler accepts only a `TokenizedRequest` carrying that
buffer and already-bounded immutable limits; it never parses or tokenizes text.

Streaming events are typed:

- Accepted: effective limits and model digest;
- Token: token ID, optional decoded delta, and position;
- Usage: input, cached-input, output, and total token counts;
- Completed: finish reason and final usage;
- Failed: bounded public error and retryability.

Internal filesystem paths, device pointers, CUDA error text, worker endpoints,
kernel cache paths, stack traces, and raw prompts must not appear in public
errors or default logs.

## 6. Execution contract

One scheduler owns request and KV state. One worker owns each accelerator.
Workers accept immutable schedule plans and return typed completion records.

A schedule plan contains:

- plan sequence and model-plan generation;
- prefill and decode row descriptors;
- token and page-table views;
- selected kernel capability IDs;
- sampling descriptors;
- completion slots and cancellation generation.

Workers do not choose admission policy, prefix eviction, or fairness. The
scheduler does not invoke HTTP handlers, inspect model-family subclasses, or
branch on CUDA architecture.

## 7. Supported v1 capability

The first release supports:

- one loaded model per engine instance;
- one CUDA GPU;
- one validated dense Llama-style decoder plan;
- BF16 weights and BF16 KV;
- BPE tokenization from tokenizer.json;
- paged full attention;
- continuous batching and chunked prefill;
- greedy, temperature, top-k, and top-p sampling;
- streaming generation and deterministic cancellation;
- OpenAI-compatible text generation at the outer API boundary.

Every other capability is unsupported unless it has a named phase gate and a
positive startup capability check.

## 8. Configuration contract

Configuration is versioned and divided into focused records:

- service: listen addresses, request-size limits, graceful drain;
- model: immutable artifact identity and architecture policy;
- device: assigned devices and memory ceiling;
- scheduler: token budget, prefill chunk, fairness policy;
- cache: page size, memory fraction, prefix policy;
- kernels: manifest path and allowed developer overrides;
- telemetry: endpoint, cardinality limits, and payload-redaction policy.

The CLI exposes only common operator choices. It does not mirror every
internal field. Automatic choices appear in a startup plan with their reason.
Unknown fields and invalid combinations fail closed.

## 9. Compatibility

The native protocol is versioned independently from OpenAI compatibility.
Breaking native changes require a new protocol version. A deprecated field is
accepted for at most one minor release unless a published compatibility
commitment says otherwise.

Model plans, kernel manifests, prefix keys, and on-disk caches carry explicit
schema versions. An incompatible artifact is rejected instead of silently
reinterpreted.

## 10. Security and privacy

- Raw prompts and generated text are absent from logs by default.
- Prefix-cache identity includes model, tokenizer, adapter, RoPE, layout, and
  relevant input digests.
- Cache sharing is opt-in and scoped; unrelated security scopes never share.
- Model files are read-only after validation.
- Runtime JIT and arbitrary executable model extensions are disabled in
  production.
- Native resources use checked sizes and explicit release.
- All externally controlled lengths are bounded before allocation.

## 11. Explicit non-goals for v1

- Fleet scheduling, tenant governance, billing, or model registry.
- Training, fine-tuning, or arbitrary graph execution.
- Python model plugins and remote code.
- Multimodal, MoE, LoRA, speculative decoding, and quantization.
- Cross-node tensor or pipeline parallelism.
- Prefill/decode disaggregation.
- A universal GPU compiler or support for every accelerator backend.
- Silent compatibility fallbacks.

## 12. Release acceptance

The first usable release must:

1. load the declared model without Python;
2. match reference logits and greedy tokens within the declared tolerance;
3. stream bounded requests and cancel them without leaked pages;
4. maintain deterministic page accounting under saturation;
5. continuously batch mixed prefill and decode requests;
6. survive malformed requests without worker corruption;
7. expose truthful TTFT, inter-token latency, queue, cache, and memory metrics;
8. pass native sanitizer and long-running leak tests;
9. produce reproducible comparisons against pinned vLLM and SGLang baselines;
10. run behind LunaNexa as an opaque runtime without either repository importing
    the other.
