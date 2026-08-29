#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
runner=scripts/run-qwen3-bf16-v12-serving-physical-campaign.sh
fixture=tests/qwen3_bf16_serving_supervisor/fixture-cli.sh

fail() {
  printf 'Qwen3 v12 serving campaign validator failed: %s\n' "$1" >&2
  exit 1
}

bash -n "$runner" || fail 'campaign syntax is invalid'
for anchor in \
  'runtime_recipe=dense_qwen3_bf16_paged_aot_v12' \
  'plan-check' 'layer_count=28' 'serve-check' 'metrics-check' \
  'network_balance=accepts1,disconnects1' \
  'kv_balance=used0,free$total_pages' \
  'drain_acknowledged=1' 'gpu_processes_during=1' \
  'engine_servers_resident_concurrently=1' \
  'authenticated-qwen-token-id-sse-benchmark-bridge-unavailable' \
  'standard_openai_responses_profile_satisfies_benchmark_adapter=false' \
  'release_bind_max_batch_rows=1' \
  'release_bind_max_query_rows=1' \
  'benchmark_c32=not-run-separate-release-profile' \
  'benchmark_c32_software_admission=present-not-exercised' \
  'benchmark_c32_release_bind_max_batch_rows=32' \
  'benchmark_c32_release_bind_max_query_rows=32' \
  'lunaflux_seal_evidence_directory'; do
  grep -Fq "$anchor" "$runner" || fail "campaign anchor is absent: $anchor"
done
if grep -E -i 'llama|mistral|vllm|sglang' "$runner" >/dev/null; then
  fail 'Qwen-only campaign names or launches another engine/model family'
fi
if bash "$runner" >/dev/null 2>&1; then
  fail 'campaign accepted missing arguments'
else
  status=$?
  [ "$status" -eq 2 ] || fail 'campaign missing-argument status drifted'
fi

target=$(mktemp -d /tmp/lunaflux-qwen3-serving-validator.XXXXXX)
cleanup() {
  chmod -R u+rwX "$target" 2>/dev/null || true
  rm -rf -- "$target"
}
trap cleanup EXIT HUP INT TERM

moon check tests/qwen3_bf16_physical tests/qwen3_bf16_serving_supervisor \
  --target native --deny-warn --warn-list +73 --target-dir "$target/build"
moon test tests/qwen3_bf16_physical --target native --deny-warn \
  --warn-list +73 --target-dir "$target/test"
moon build tests/qwen3_bf16_serving_supervisor --target native --release \
  --deny-warn --warn-list +73 --target-dir "$target/supervisor-build"
supervisor=$target/supervisor-build/native/release/build/tests/qwen3_bf16_serving_supervisor/qwen3_bf16_serving_supervisor.exe
[ -x "$supervisor" ] || fail 'supervisor executable is absent'
[ -x "$fixture" ] || fail 'supervisor fixture is not executable'

runtime_stdout=$target/runtime.stdout
runtime_stderr=$target/runtime.stderr
trigger=$target/drain.request
"$supervisor" "$repo_root/$fixture" \
  '/tmp/qwen-release#sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  "$runtime_stdout" "$runtime_stderr" "$trigger" \
  >"$target/supervisor.stdout" 2>"$target/supervisor.stderr" &
supervisor_pid=$!
ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if [ -f "$runtime_stdout" ] &&
    grep -Fxq 'readiness: true' "$runtime_stdout"; then
    ready=1
    break
  fi
  sleep 0.1
done
[ "$ready" -eq 1 ] || fail 'supervisor fixture did not publish readiness'
: >"$trigger"
wait "$supervisor_pid" || fail 'supervisor fixture lifecycle failed'
[ ! -s "$runtime_stderr" ] && [ ! -s "$target/supervisor.stderr" ] ||
  fail 'supervisor fixture emitted stderr'
for exact in \
  'schema=lunaflux-qwen3-native-supervisor.v1' \
  'supervisor_status=0' 'drain_acknowledged=1' 'child_exit_code=0' \
  'child_closed=1'; do
  grep -Fxq "$exact" "$target/supervisor.stdout" ||
    fail "supervisor fixture evidence lost: $exact"
done
if "$supervisor" "$repo_root/$fixture" \
  '/tmp/qwen-release#sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  "$runtime_stdout" "$runtime_stderr" "$trigger" >/dev/null 2>&1; then
  fail 'supervisor overwrote existing runtime evidence'
fi
if "$supervisor" >/dev/null 2>&1; then
  fail 'supervisor accepted missing arguments'
else
  status=$?
  [ "$status" -eq 2 ] || fail 'supervisor missing-argument status drifted'
fi

sh -n "$fixture"
git diff --check
printf '%s\n' 'Qwen3 v12 serving physical campaign static validation passed'
