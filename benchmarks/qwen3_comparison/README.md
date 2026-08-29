# Qwen3-0.6B comparative benchmark

This external harness compares only LunaFlux, vLLM, and SGLang on the pinned
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

The three servers must already be persistent and healthy before the campaign
starts. Warmup requests are issued for every engine/profile and excluded from
measurement. Each prefill/decode profile runs at concurrency 1, 8, and 32 in
three fixed Latin-square rounds:

1. LunaFlux, vLLM, SGLang;
2. vLLM, SGLang, LunaFlux;
3. SGLang, LunaFlux, vLLM.

The hardware-capacity declaration must admit concurrency 32 or the campaign
fails before measurement.

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

## Persistent baseline commands

No environment is created or modified. The named environments and package
versions must already exist:

```sh
scripts/start-qwen3-vllm-benchmark-server.sh \
  ABS_PINNED_VLLM_CONDA_ENV_PREFIX EXACT_VLLM_VERSION ABS_PINNED_QWEN3_ROOT \
  ABS_MODEL_INVENTORY#sha256=MODEL_INVENTORY_SHA256 127.0.0.1 8101
```

```sh
scripts/start-qwen3-sglang-benchmark-server.sh \
  ABS_PINNED_SGLANG_CONDA_ENV_PREFIX EXACT_SGLANG_VERSION ABS_PINNED_QWEN3_ROOT \
  ABS_MODEL_INVENTORY#sha256=MODEL_INVENTORY_SHA256 127.0.0.1 8102
```

LunaFlux must expose the canonical token-ID SSE benchmark bridge declared in
`campaign.template.json`. That bridge is currently the remaining serving-step
integration requirement; the harness does not substitute a text prompt or a
different model when it is absent.

## Running

Fill and hash `config/campaign.template.json`, then execute:

```sh
python3 -B benchmarks/qwen3_comparison/campaign.py \
  --campaign ABS_CAMPAIGN_JSON#sha256=CAMPAIGN_SHA256 \
  --workload ABS_TOKENIZED_WORKLOAD_JSONL#sha256=WORKLOAD_SHA256 \
  --output ABS_NEW_RESULT_DIRECTORY
```

The output contains raw per-request JSONL, trial JSONL, correctness joins, and
per-engine/profile summaries for TTFT, inter-token latency, E2E latency,
request throughput, output-token throughput, error rate, and whole-device GPU
memory. Summaries contain median, p95, and deterministic bootstrap 95% median
confidence intervals. They do not select a winner. Any incomplete or unequal
greedy-output hash, per-token streaming, output-count consistency, or complete
request set sets `speed_comparison_valid=false`; the latency and throughput
numbers then remain descriptive measurements only.

No result about Ollama may be inferred from either vLLM or SGLang. An Ollama
comparison requires its own measured, pinned campaign and is explicitly
outside this harness.

The complete static and hostile test invocation is:

```sh
python3 -B -m unittest discover -s benchmarks/qwen3_comparison -p 'test_*.py'
```
