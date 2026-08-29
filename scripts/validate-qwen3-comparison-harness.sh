#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
package=benchmarks/qwen3_comparison

fail() {
  printf 'Qwen3 comparison harness validation failed: %s\n' "$1" >&2
  exit 1
}

sh -n scripts/start-qwen3-vllm-benchmark-server.sh
sh -n scripts/start-qwen3-sglang-benchmark-server.sh
sh -n scripts/start-qwen3-lunaflux-token-id-bridge.sh
sh -n scripts/start-qwen3-lunaflux-benchmark-server.sh
python3 -B -m unittest \
  benchmarks.qwen3_comparison.test_campaign \
  benchmarks.qwen3_comparison.test_adapters \
  benchmarks.qwen3_comparison.test_lifecycle
python3 -B -m unittest discover -s benchmarks/qwen3_comparison -p 'test_*.py'

for anchor in \
  'Qwen3-0.6B' \
  'input_token_ids_sha256' \
  'validate_model_inventory' \
  'model_inventory_full_scan_count' \
  'model-admission.json' \
  'one-engine-per-target-gpu-coordinate' \
  'require_clean_gpu' \
  'CUDA_VISIBLE_DEVICES' \
  'start_new_session=True' \
  'process_group_leader_command_sha256' \
  'authenticated_capacity_receipt' \
  'prefix_reuse' \
  'scheduler_policy' \
  'diagnostic-token-id-sse-bridge-only' \
  'warmup_excluded' \
  'latin_square_order' \
  'ttft_millis' \
  'inter_token_latency_millis' \
  'request_throughput_per_second' \
  'output_token_throughput_per_second' \
  'gpu_memory_used_peak_mib' \
  'output_token_ids_sha256' \
  'return_token_ids' \
  'output_ids' \
  'stream_interval' \
  'ignore_eos' \
  'lunaflux_lifecycle' \
  'owned_executables' \
  'token_timing_exact' \
  'output_count_consistent' \
  'median_ci95' \
  'speed_comparison_valid' \
  'correctness_failure_invalidates_speed_comparison' \
  'forbidden: no Ollama result may be inferred'; do
  rg -Fq "$anchor" "$package" || fail "required contract anchor is absent: $anchor"
done
if rg -Fq '_append_retokenized_timestamps' "$package"; then
  fail 'baseline adapter retained text retokenization as a token-ID fallback'
fi

for anchor in \
  'native runtime executable' \
  'native supervisor executable' \
  'token-ID bridge executable' \
  'runtime_origin=luna+tcp://' \
  'drain_trigger'; do
  rg -Fq "$anchor" scripts/start-qwen3-lunaflux-benchmark-server.sh ||
    fail "combined LunaFlux lifecycle anchor is absent: $anchor"
done
if rg -Fq 'python3' scripts/start-qwen3-lunaflux-token-id-bridge.sh; then
  fail 'production token-ID bridge launcher acquired a Python dependency'
fi
rg -Fq 'verify-qwen3-authenticated-capacity.sh' \
  scripts/start-qwen3-lunaflux-benchmark-server.sh ||
  fail 'combined LunaFlux lifecycle does not verify exact c32 authority'

for server in \
  scripts/start-qwen3-vllm-benchmark-server.sh \
  scripts/start-qwen3-sglang-benchmark-server.sh; do
  rg -Fq 'unset PYTHONHOME PYTHONPATH PYTHONSTARTUP' "$server" ||
    fail "baseline launcher inherits unrelated Python environments: $server"
  rg -Fq 'PATH=$environment/bin:/usr/bin:/bin' "$server" ||
    fail "baseline launcher cannot find environment-local runtime tools: $server"
  rg -Fq '"$environment/bin/python"' "$server" || fail "server does not use the absolute pinned Conda environment: $server"
  if rg -Fq 'conda run' "$server"; then
    fail "server requires unsupported conda run: $server"
  fi
  rg -Fq '"model_type"' "$server" || fail "server does not reject non-Qwen config: $server"
  rg -Fq 'verify_model_admission.py' "$server" || fail "server does not consume campaign model admission: $server"
  if rg -Fq 'verify_model_inventory.py' "$server"; then
    fail "server rescans the large model inventory on every restart: $server"
  fi
  if rg -n 'pip install|conda install|modelscope download|huggingface-cli download' "$server"; then
    fail "server command installs or downloads dependencies: $server"
  fi
  if "$server" >/dev/null 2>&1; then
    fail "server command accepted a missing absolute environment prefix: $server"
  fi
done
rg -Fq -- '--no-enable-prefix-caching' scripts/start-qwen3-vllm-benchmark-server.sh ||
  fail 'vLLM launcher does not disable prefix caching'
rg -Fq -- '--stream-interval 1' scripts/start-qwen3-vllm-benchmark-server.sh ||
  fail 'vLLM launcher does not expose one-token stream timing'
rg -Fq -- '--disable-radix-cache' scripts/start-qwen3-sglang-benchmark-server.sh ||
  fail 'SGLang launcher does not disable radix/prefix caching'
rg -Fq -- '--random-seed 0' scripts/start-qwen3-sglang-benchmark-server.sh ||
  fail 'SGLang launcher does not bind its process-level random seed'
rg -Fq -- '--skip-tokenizer-init' scripts/start-qwen3-sglang-benchmark-server.sh ||
  fail 'SGLang launcher does not expose exact streamed output token IDs'
rg -Fq -- '--stream-interval 1' scripts/start-qwen3-sglang-benchmark-server.sh ||
  fail 'SGLang launcher does not expose one-token stream timing'
rg -Fq -- '--scheduling-policy fcfs' scripts/start-qwen3-vllm-benchmark-server.sh ||
  fail 'vLLM launcher does not bind FCFS scheduling'
rg -Fq -- '--schedule-policy fcfs' scripts/start-qwen3-sglang-benchmark-server.sh ||
  fail 'SGLang launcher does not bind FCFS scheduling'
[ -f "$package/verify_model_inventory.py" ] || fail 'exact source-model inventory verifier is absent'
[ -f "$package/verify_model_admission.py" ] || fail 'campaign-local model admission verifier is absent'
if python3 -B "$package/verify_model_inventory.py" >/dev/null 2>&1; then
  fail 'source-model inventory verifier accepted missing arguments'
fi
if python3 -B "$package/verify_model_admission.py" >/dev/null 2>&1; then
  fail 'model admission verifier accepted missing arguments'
fi

if rg -n 'shell[[:space:]]*=[[:space:]]*True|os\.system|subprocess\.call' \
  "$package"; then
  fail 'benchmark lifecycle permits arbitrary shell command execution'
fi
if rg -Fq 'whole-device-with-all-persistent-servers-resident' "$package"; then
  fail 'benchmark still contaminates measurements with three resident servers'
fi

if rg -n -i '\bllama\b' "$package" scripts/start-qwen3-vllm-benchmark-server.sh \
  scripts/start-qwen3-sglang-benchmark-server.sh; then
  fail 'Qwen-only benchmark contains another model family'
fi
if rg -n 'qwen3_comparison' engine model ops runtime release cmd --glob '*.mbt' --glob 'moon.pkg'; then
  fail 'external benchmark leaked into the production request path'
fi
if python3 -B "$package/campaign.py" >/dev/null 2>&1; then
  fail 'campaign accepted missing pinned inputs'
fi

printf '%s\n' 'Qwen3 comparison harness static validation passed'
