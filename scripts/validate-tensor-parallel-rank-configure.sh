#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package="engine/tensor_parallel_rank_configure"
rank_package="engine/rank_group_process"

for required in \
  "$package/moon.pkg" \
  "$package/types.mbt" \
  "$package/codec.mbt" \
  "$package/topology_codec.mbt" \
  "$package/authenticate.mbt" \
  "$package/frame_buffer.mbt"; do
  if [ ! -f "$required" ]; then
    printf 'rank Configure boundary file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'cuda|nccl|device_executor|device_memory|numeric|model/plan|model-family' \
    "$package" "$package/moon.pkg" --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'rank Configure owner crossed into execution/backend policy' >&2
  exit 1
fi
if ! rg -q 'priv storage : FixedArray\[Byte\]' "$package/types.mbt" ||
  rg -n 'priv .*: Bytes|configure_payload|opaque.*payload' \
    "$package" "$rank_package/types.mbt" "$rank_package/prepare.mbt" \
    >/dev/null; then
  printf '%s\n' 'rank Configure authority is not bounded typed storage' >&2
  exit 1
fi
if ! rg -q \
    'priv configure : @tensor_parallel_rank_configure.TensorParallelRankConfigureContract' \
    "$rank_package/types.mbt" ||
  rg -n 'RankGroupProcessRankStartup::new\([^)]*,|configure_payload : Bytes' \
    "$rank_package" --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'rank-group process still admits an arbitrary Configure blob' >&2
  exit 1
fi

for evidence in \
  RANK_CONFIGURE_MAGIC \
  RANK_CONFIGURE_VERSION \
  RANK_CONFIGURE_CHECKSUM_OFFSET \
  RANK_CONFIGURE_DIGEST_OFFSET \
  RANK_CONFIGURE_CAPACITY \
  SELECTED_TOPOLOGY_MAX_BYTES \
  ValidatedTensorParallelRankConfigureFrame \
  StaleFrame; do
  if ! rg -q "$evidence" "$package" --glob '*.mbt'; then
    printf 'rank Configure framing evidence is missing: %s\n' "$evidence" >&2
    exit 1
  fi
done

for nested in \
  StartupFrameBuffer \
  decode_bootstrap_source \
  decode_selected_topology \
  TensorParallelRankEnvelopeFrameBuffer; do
  if ! rg -q "$nested" "$package/codec.mbt"; then
    printf 'rank Configure nested codec is missing: %s\n' "$nested" >&2
    exit 1
  fi
done

for claim in \
  ModelIdentity \
  ModelGeneration \
  GroupGeneration \
  Rank \
  WorldSize \
  DeviceOrdinal \
  BootstrapSource \
  worker_contract_digest \
  group_digest \
  topology_digest \
  device_plan_digest \
  execution_plan_digest \
  collective_digest \
  kv_digest; do
  if ! rg -q "$claim" "$package/authenticate.mbt"; then
    printf 'rank Configure cross-auth evidence is missing: %s\n' "$claim" >&2
    exit 1
  fi
done
if ! rg -q 'predecessor_value' "$package/codec.mbt" ||
  ! rg -q 'previous_plan_sequence_value' "$rank_package/prepare.mbt"; then
  printf '%s\n' 'rank Configure predecessor authentication is missing' >&2
  exit 1
fi

for hostile in \
  corruption \
  reserved \
  replay \
  source \
  ordinal \
  envelope \
  execution-plan \
  stale; do
  if ! rg -qi "$hostile" "$package/configure_wbtest.mbt"; then
    printf 'rank Configure hostile test is missing: %s\n' "$hostile" >&2
    exit 1
  fi
done

for file in "$package"/*.mbt "$rank_package"/*.mbt; do
  if [ "$(wc -l < "$file")" -gt 500 ]; then
    printf '%s exceeds 500 lines\n' "$file" >&2
    exit 1
  fi
done

if [ "${1:-}" = "--static-only" ]; then
  printf '%s\n' 'LunaFlux typed rank Configure static boundary passed.'
  exit 0
fi

moon test "$package" --target native --release --deny-warn
moon test "$rank_package" --target native --release --deny-warn

configure_c="_build/native/release/test/engine/tensor_parallel_rank_configure/tensor_parallel_rank_configure.whitebox_test.c"
rank_c="_build/native/release/test/engine/rank_group_process/rank_group_process.whitebox_test.c"
if [ ! -f "$configure_c" ] || [ ! -f "$rank_c" ]; then
  printf '%s\n' 'rank Configure release allocation evidence is missing' >&2
  exit 1
fi

extract_definition() {
  local source="$1"
  local symbol="$2"
  awk -v symbol="$symbol" '
    index($0, symbol) > 0 &&
      $0 ~ /^(struct|int|uint|void|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ {
      candidate = 1; body = $0 ORS; next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) {
        copying = 1; depth = 1; printf "%s", body; candidate = 0; next
      }
    }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "$source"
}

hot=""
for item in \
  "$configure_c|TensorParallelRankConfigureContract8copy__to(" \
  "$configure_c|ValidatedTensorParallelRankConfigureFrame8copy__to("; do
  source="${item%%|*}"
  symbol="${item#*|}"
  body="$(extract_definition "$source" "$symbol")"
  if [ -z "$body" ]; then
    printf 'rank Configure allocation symbol is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot="${hot}${body}"
done
if printf '%s\n' "$hot" | rg -q 'moonbit_make_|moonbit_add_string' ||
  printf '%s\n' "$hot" | rg 'moonbit_malloc' |
    rg -vq 'TensorParallelRankConfigureError'; then
  printf '%s\n' 'rank Configure publication path allocates' >&2
  exit 1
fi
if ! rg -q 'moonbit_make_bytes' "$configure_c"; then
  printf '%s\n' 'rank Configure allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux typed rank Configure boundary/allocation gate passed.'
