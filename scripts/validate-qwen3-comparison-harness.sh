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
python3 -B -m unittest \
  benchmarks.qwen3_comparison.test_campaign \
  benchmarks.qwen3_comparison.test_adapters
python3 -B -m unittest discover -s benchmarks/qwen3_comparison -p 'test_*.py'

for anchor in \
  'Qwen3-0.6B' \
  'input_token_ids_sha256' \
  'validate_model_inventory' \
  'warmup_excluded' \
  'latin_square_order' \
  'ttft_millis' \
  'inter_token_latency_millis' \
  'request_throughput_per_second' \
  'output_token_throughput_per_second' \
  'gpu_memory_used_peak_mib' \
  'output_token_ids_sha256' \
  'token_timing_exact' \
  'output_count_consistent' \
  'median_ci95' \
  'speed_comparison_valid' \
  'correctness_failure_invalidates_speed_comparison' \
  'forbidden: no Ollama result may be inferred'; do
  rg -Fq "$anchor" "$package" || fail "required contract anchor is absent: $anchor"
done

for server in \
  scripts/start-qwen3-vllm-benchmark-server.sh \
  scripts/start-qwen3-sglang-benchmark-server.sh; do
  rg -Fq '"$environment/bin/python"' "$server" || fail "server does not use the absolute pinned Conda environment: $server"
  if rg -Fq 'conda run' "$server"; then
    fail "server requires unsupported conda run: $server"
  fi
  rg -Fq '"model_type"' "$server" || fail "server does not reject non-Qwen config: $server"
  rg -Fq 'verify_model_inventory.py' "$server" || fail "server does not authenticate the exact model file set: $server"
  if rg -n 'pip install|conda install|modelscope download|huggingface-cli download' "$server"; then
    fail "server command installs or downloads dependencies: $server"
  fi
  if "$server" >/dev/null 2>&1; then
    fail "server command accepted a missing absolute environment prefix: $server"
  fi
done
[ -f "$package/verify_model_inventory.py" ] || fail 'exact source-model inventory verifier is absent'
if python3 -B "$package/verify_model_inventory.py" >/dev/null 2>&1; then
  fail 'source-model inventory verifier accepted missing arguments'
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
