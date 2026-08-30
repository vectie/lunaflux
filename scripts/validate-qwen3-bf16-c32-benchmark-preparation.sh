#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
campaign=$repo_root/scripts/run-qwen3-bf16-c32-benchmark-preparation.sh
common=$repo_root/scripts/qwen3-c32-preparation-common.sh
materializer=$repo_root/scripts/materialize-qwen3-bf16-v12-launch.sh
fail() {
  printf 'Qwen3 c32 benchmark preparation validation rejected: %s\n' "$1" >&2
  exit 1
}

bash -n "$campaign" "$common" "$materializer" ||
  fail 'campaign/helper syntax is invalid'
set +e
usage_output=$("$campaign" 2>&1)
usage_status=$?
set -e
[ "$usage_status" -eq 2 ] || fail 'missing arguments do not fail with usage status'
printf '%s\n' "$usage_output" | grep -Fq 'ABSOLUTE_NEW_ARTIFACT_ROOT' ||
  fail 'usage omits the distinct artifact root'

for exact in \
  '13 1 115 8 8192 32 256 8192 "$candidate"' \
  '"$toolchain_manifest" "$toolchain_sha" 13 1 115 8 8192 32 256 8192' \
  'native-framed-c32-benchmark-v1' \
  'materialize-qwen3-authenticated-capacity.sh' \
  'verify-qwen3-authenticated-capacity.sh' \
  'Qwen c32 benchmark requires authenticated device greedy sampling' \
  'sampling_runtime=embedded_cuda_greedy_v1' \
  'max_plan_pages=$max_page_table_entries' \
  'block_table_pages_per_request\":$context_page_table_entries' \
  'for request_id in $(seq 1 32)' \
  'event_sequence == '\''token:92648,token:4532,terminal,done'\''' \
  'export CUDA_VISIBLE_DEVICES="$expected_uuid"' \
  'kill -TERM "$pid"' \
  'kill -KILL -- "-$pgid"' \
  'benchmark_performance_claim=not-made'; do
  grep -Fq "$exact" "$campaign" "$common" "$materializer" ||
    fail "campaign boundary is absent: $exact"
done

if grep -Fq 'max_graph_capture_bytes' "$materializer"; then
  fail 'Qwen eager-only materializer must not request graph memory'
fi

if grep -Fq 'kill -TERM -- "-$pgid"' "$campaign" "$common"; then
  fail 'normal cleanup must not TERM the whole owned process group'
fi

if grep -Fq 'Qwen c32 CLI build emitted stderr' "$campaign"; then
  fail 'successful Moon builds must not be rejected for progress on stderr'
fi
grep -Fq 'status plus --deny-warn is the admission boundary' "$campaign" ||
  fail 'Moon build stderr preservation rationale is absent'

for forbidden in \
  'run-qwen3-bf16-physical-campaign.sh' \
  'run-qwen3-bf16-v12-serving-physical-campaign.sh' \
  'native-framed-v1 c1' \
  '16 8 1 1 8'; do
  if grep -Fq "$forbidden" "$campaign"; then
    fail "campaign reuses a lower/c1 boundary: $forbidden"
  fi
done

grep -Fq 'authentication_work=request_path=0,startup_offline_only=1' "$campaign" ||
  fail 'startup-only authentication boundary is absent'
grep -Fq 'all_requests_matched_qwen_greedy_prefix=true' "$campaign" ||
  fail 'c32 correctness result is absent'
"$repo_root/scripts/validate-qwen3-capacity-receipt.sh" >/dev/null ||
  fail 'authenticated capacity receipt hostile validation failed'
printf '%s\n' 'Qwen3 BF16 c32 benchmark preparation validation passed.'
