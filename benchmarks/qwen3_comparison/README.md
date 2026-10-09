# Qwen3-0.6B comparative benchmark

This external harness compares LunaFlux, vLLM, SGLang, and llama.cpp on the pinned
dense Qwen3-0.6B model. It is not imported by the serving runtime, does not
install packages, does not download models, and does not contain a benchmark
result.

All engines receive the same pre-rendered Qwen chat input token IDs, output
token limit, and greedy sampling declaration. The corpus preparation command
loads only local model files, authenticates `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and the Qwen chat template, and cross-checks the
text-rendering and direct-tokenization paths. The source-model inventory and
the LunaFlux converted numeric artifact/route identities are independently
bound in the campaign.

Servers must not be prestarted. For every engine in each Latin-square trial the
harness admits a clean target GPU, launches exactly one engine with a digest-pinned
absolute launcher, verifies its exact package, process group, executable
identities, model, and
health identity, performs excluded warmups, measures every token-shape and
concurrency coordinate, drains
and terminates the owned process group, waits for cooldown, and admits a clean
GPU again. Startup, readiness, warmup, drain, shutdown, and cooldown time are
all outside the measured interval. Schema v2 measures the fixed `(input,
output)` token vector `(59,256)`, `(128,128)`, `(512,64)`, and `(1528,32)`;
each shape runs at concurrency 1, 8, and 32. This makes token counts a vector
rather than a single favorable point. Schema v2 uses four fixed Latin-square
rounds:

1. LunaFlux, vLLM, SGLang, llama.cpp;
2. vLLM, SGLang, llama.cpp, LunaFlux;
3. SGLang, llama.cpp, LunaFlux, vLLM;
4. llama.cpp, LunaFlux, vLLM, SGLang.

Schema v1 remains available for archived three-engine campaigns.

The child environment binds `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES` to the same GPU UUID that
the lifecycle and memory sampler inspect. All baseline adapters consume the
engines' exact streamed output-token IDs: vLLM uses
`return_token_ids=true`, while SGLang runs with `--skip-tokenizer-init` and
consumes `/generate`'s cumulative `output_ids` using its native
server-level `--random-seed 0` spelling. Its request binds every sampling
parameter explicitly, including greedy `top_k=1`, neutral penalties, no stop
constraints, `ignore_eos=true`, and a one-token stream interval; model
generation defaults cannot enter the workload. Text retokenization is never
used as a substitute for correctness. The exact output IDs from every engine are decoded once by
the same pinned Qwen tokenizer after the measured response completes. Both
launchers bind a one-token stream
interval so per-token timing fails closed if an engine batches token events.
llama.cpp receives the same prompt token IDs through `/completion`, returns raw
streamed `tokens`, and sets `cache_prompt=false`; it runs the unquantized BF16
GGUF conversion with continuous batching, Flash Attention, BF16 KV cache, 32
slots, and a 65,536-token shared context (2,048 tokens per slot). Each
configured warmup is a complete round at the profile's declared
concurrency, rather than a sequential request that leaves c8/c32 batching and
CUDA-graph paths cold.
All four adapters also bind `ignore_eos=true`; the prefill and decode profiles
therefore measure the same fixed 32- and 256-token continuations rather than
engine-specific EOS stopping behavior.
vLLM is launched with `--generation-config vllm`; SGLang receives the same
neutral greedy policy explicitly in every native `/generate` request.

The hardware-capacity declaration must admit concurrency 32. LunaFlux must
also supply a digest-bound authenticated capacity receipt for the exact model,
configuration, runtime executable, and diagnostic token-ID bridge with
`max_concurrency >= 32`; otherwise the campaign fails before measurement.
Prefix reuse is disabled for every engine; all four descriptors
bind FCFS scheduling, automatic KV-cache dtype, and a 32-sequence ceiling.
Those policy fields are copied into every trial and lifecycle record so a
default or flag drift cannot silently enter a speed comparison.

The exact model inventory, including the large weight files, is authenticated
once during campaign preparation. The harness then writes a small canonical
campaign-local `model-admission.json` receipt. Every lifecycle launch consumes
that receipt and its digest plus the immutable canonical model path, so model
weights are not re-hashed on each restart. The receipt and its creation are
outside measured work.

## Preparing exact inputs

Create at least 32 unique `prefill` and 32 unique `decode` message rows using
`config/messages.template.jsonl`, then run from the repository root:

```sh
python3 -B benchmarks/qwen3_comparison/prepare_corpus.py \
  --model-root ABS_PINNED_QWEN3_ROOT \
  --config-sha256 CONFIG_SHA256 \
  --tokenizer-json-sha256 TOKENIZER_JSON_SHA256 \
  --tokenizer-config-sha256 TOKENIZER_CONFIG_SHA256 \
  --chat-template-sha256 CHAT_TEMPLATE_SHA256 \
  --messages ABS_MESSAGES_JSONL#sha256=MESSAGES_SHA256 \
  --output ABS_NEW_TOKENIZED_WORKLOAD_JSONL
```

## Lifecycle launcher contract

No environment is created or modified. The campaign calls each absolute,
digest-pinned launcher directly without a shell command string. The vLLM and
SGLang launchers use this fixed argument contract:

```text
LAUNCHER ENV_PREFIX_OR_NATIVE EXACT_VERSION ABS_MODEL_ROOT ABS_MODEL_ADMISSION#sha256=HEX 127.0.0.1 PORT
```

The vLLM and SGLang implementations are:

```sh
scripts/start-qwen3-vllm-benchmark-server.sh \
  ABS_PINNED_VLLM_CONDA_ENV_PREFIX EXACT_VLLM_VERSION ABS_PINNED_QWEN3_ROOT \
  ABS_MODEL_ADMISSION#sha256=MODEL_ADMISSION_SHA256 127.0.0.1 8101
```

```sh
scripts/start-qwen3-sglang-benchmark-server.sh \
  ABS_PINNED_SGLANG_CONDA_ENV_PREFIX EXACT_SGLANG_VERSION ABS_PINNED_QWEN3_ROOT \
  ABS_MODEL_ADMISSION#sha256=MODEL_ADMISSION_SHA256 127.0.0.1 8102
```

These examples show the strict launcher interface; operators do not start them
alongside the campaign. The orchestrator starts and stops them one at a time.
Their `revision_sha256` fields are required to be `null`: a wheel distribution
version is not a source revision. Reproducibility instead binds the observed
installed distribution version, launcher and Python executable hashes, fixed
CLI arguments, model inventory, GPU identity, and workload. LunaFlux alone
binds its real lowercase source revision SHA-256 in both `revision_sha256` and
`package_version`.

llama.cpp uses the same first six lifecycle fields plus the exact BF16 GGUF
SHA-256. Its root must contain `build-cuda/bin/llama-server` and
`models/Qwen3-0.6B-BF16.gguf`; both are digest-bound by the campaign:

```sh
scripts/start-qwen3-llama-cpp-benchmark-server.sh \
  ABS_LLAMA_CPP_ROOT EXACT_LLAMA_CPP_VERSION ABS_PINNED_QWEN3_ROOT \
  ABS_MODEL_ADMISSION#sha256=MODEL_ADMISSION_SHA256 127.0.0.1 8103 \
  EXACT_BF16_GGUF_SHA256
```

LunaFlux uses the same first six lifecycle fields followed by the digest-bound
native runtime, supervisor, token-ID bridge, deployment, tokenizer, launch,
release binding, capacity receipt, native address, model identities, and
token/context limits
declared by `lunaflux_lifecycle`. Its combined launcher remains the process
group leader: it starts the supervisor/native runtime, waits for the exact
native readiness and runtime origin, then starts the bridge. A TERM to the
leader first closes the bridge and then asks the native supervisor to drain.
The lifecycle record independently finds and hashes the runtime, supervisor,
and bridge executables among that owned process group's live children. Thus a
prestarted native service cannot bypass the clean-GPU admission or be mistaken
for the measured engine.
Before opening the GPU, the benchmark launcher reconstructs and verifies the
capacity receipt from the exact c32 release binding, native-framed deployment,
runtime executable, and bridge executable. A free-form authentication digest
is not accepted.

This orchestration lives only in
`scripts/start-qwen3-lunaflux-benchmark-server.sh`. The production-facing
`start-qwen3-lunaflux-token-id-bridge.sh` remains a pure 12-argument executable
handoff with no Python dependency.

The standard text-only `/v1/responses` endpoint is explicitly rejected because
it does not accept the custom exact `input_token_ids` protocol. The harness
does not substitute a text prompt or a different model when the bridge is
absent.

## Running

Fill and hash `config/campaign.template.json`, then execute:

```sh
python3 -B benchmarks/qwen3_comparison/campaign.py \
  --campaign ABS_CAMPAIGN_JSON#sha256=CAMPAIGN_SHA256 \
  --workload ABS_TOKENIZED_WORKLOAD_JSONL#sha256=WORKLOAD_SHA256 \
  --output ABS_NEW_RESULT_DIRECTORY
```

The output contains raw per-request JSONL, lifecycle JSONL with exact package,
PID/process-group, executable, command-line, launcher and clean-GPU identity,
trial JSONL, correctness joins, and
per-engine/profile summaries for TTFT, inter-token latency, E2E latency,
request throughput, output-token throughput, error rate, and whole-device GPU
memory. Summaries contain median, p95, and deterministic bootstrap 95% median
confidence intervals. They do not select a winner. Any incomplete or unequal
greedy-output hash, inconsistent output count, or incomplete request set sets
`speed_comparison_valid=false`; the latency and throughput numbers then remain
descriptive measurements only.

Servers may occasionally coalesce multiple exact output token IDs into one SSE
event even with a one-token stream interval. This does not invalidate TTFT,
E2E, output count, token IDs, or aggregate throughput. Such requests are marked
`token_timing_exact=false` and excluded only from the ITL distribution. ITL is
summarized as one median per exact-timing request before deterministic bootstrap,
so correlated token intervals from one request are not treated as independent
samples.

No result about Ollama may be inferred from any measured engine. An Ollama
comparison requires its own measured, pinned campaign and is explicitly
outside this harness.

The complete static and hostile test invocation is:

```sh
python3 -B -m unittest \
  benchmarks.qwen3_comparison.test_campaign \
  benchmarks.qwen3_comparison.test_adapters \
  benchmarks.qwen3_comparison.test_lifecycle
python3 -B -m unittest discover -s benchmarks/qwen3_comparison -p 'test_*.py'
scripts/validate-qwen3-comparison-harness.sh
```
