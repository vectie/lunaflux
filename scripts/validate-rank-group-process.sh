#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package="engine/rank_group_process"
manifest="$package/moon.pkg"
interface="$package/pkg.generated.mbti"
worker_wire_interface="engine/worker_wire/pkg.generated.mbti"

if sed -n '1,/^}/p' "$manifest" |
  rg -n 'scheduler|service|internal/(cuda|nccl)|kernel|model-family|worker_process' \
    >/dev/null; then
  printf '%s\n' 'rank-group process crossed a forbidden package boundary' >&2
  exit 1
fi
if rg -n 'Scripted|Fake' "$package" --glob '*.mbt' --glob '!**/*_test.mbt' \
    --glob '!**/*_wbtest.mbt' >/dev/null; then
  printf '%s\n' 'rank-group process contains a production test double' >&2
  exit 1
fi
if rg -n 'WorkerApprovedRoots|PreparedWorkerApprovedRoots' "$package" \
    --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'rank-group process retained approved-root concrete ownership' >&2
  exit 1
fi
if rg -n 'WorkerApprovedRoots|PreparedWorkerApprovedRoots' "$interface" \
    >/dev/null; then
  printf '%s\n' 'rank-group process leaked approved-root concrete owners' >&2
  exit 1
fi
if ! grep -Fqx \
  'pub fn PreparedRankGroupProcess::spawn_with_approved_root_spawn_authority(Self, @approved_fs_inheritance.WorkerApprovedRootSpawnAuthority) -> RankGroupProcessPreparation raise RankGroupProcessError' \
  "$interface"; then
  printf '%s\n' 'rank-group process spawn-only root authority API drifted' >&2
  exit 1
fi
if ! grep -Fq \
    'pub fn RankGroupProcessSupervisor::progress(Self, @worker_protocol.SubmittedSchedulePlan)' \
    "$interface" ||
  ! grep -Fqx \
    'pub fn PlanFrameBuffer::current(Self) -> ValidatedPlanFrame raise WorkerWireError' \
    "$worker_wire_interface"; then
  printf '%s\n' 'rank-group caller reauthentication API drifted' >&2
  exit 1
fi
if ! grep -Fqx \
  'pub fn ReceivedRankGroupCompletion::graph_telemetry(Self, RankGroupProcessSupervisor) -> @worker_wire.WorkerGraphTelemetry raise RankGroupProcessError' \
  "$interface"; then
  printf '%s\n' 'rank-group logical graph telemetry API drifted' >&2
  exit 1
fi
for field in tx rx rx_staging; do
  if [ "$(rg -c "priv ${field} : FixedArray\[Byte\]" "$package/types.mbt")" -ne 1 ]; then
    printf 'rank-group process shared buffer field drifted: %s\n' "$field" >&2
    exit 1
  fi
done
if rg -n 'tx_wire|rx_wire|graph_rx|priv payload : FixedArray\[Byte\]' "$package"/*.mbt \
    >/dev/null; then
  printf '%s\n' 'rank-group process regained a duplicate wire arena' >&2
  exit 1
fi
if rg -n 'Array::|Map::|HashMap::|FixedArray::make' \
    "$package/exchange.mbt" "$package/completion.mbt" >/dev/null; then
  printf '%s\n' 'rank-group warmed exchange path contains allocation syntax' >&2
  exit 1
fi
if rg -n \
    'Option\[(@worker_protocol.SubmittedSchedulePlan|@rank_group_protocol.SubmittedRankGroupPlan|@rank_group_protocol.CompletedRankGroupPlan)\]' \
    "$package"/*.mbt >/dev/null; then
  printf '%s\n' 'rank-group supervisor retained a success capability wrapper' >&2
  exit 1
fi

moon test "$package" --target native --release --deny-warn
generated_c="_build/native/release/test/engine/rank_group_process/rank_group_process.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'rank-group process release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^struct moonbit_result_|^int32_t |^int64_t |^void / {
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
  ' "$generated_c"
}

hot_body=""
for symbol in \
  'RankGroupProcessSupervisor5begin(' \
  'RankGroupProcessSupervisor8progress(' \
  'RankGroupProcessSupervisor20received__completion(' \
  'RankGroupProcessSupervisor18accept__completion(' \
  'RankGroupProcessSupervisor32retire__after__scheduler__commit(' \
  'RankGroupProcessSupervisor32prepare__active__frame__deadline(' \
  'RankGroupProcessSupervisor20reauthenticate__plan(' \
  'RankGroupProcessSupervisor29stage__active__encoded__write(' \
  'RankGroupProcessSupervisor13stage__submit(' \
  'RankGroupProcessSupervisor11stage__poll(' \
  'RankGroupProcessSupervisor22next__unfinished__rank(' \
  'RankGroupProcessSupervisor25all__rank__results__ready(' \
  'RankGroupProcessSupervisor12active__slot(' \
  'RankGroupProcessSupervisor29authenticate__active__scalars(' \
  'RankGroupProcessSupervisor15slot__for__plan(' \
  'RankGroupProcessSupervisor16accept__response(' \
  'RankGroupProcessSupervisor42accept__graph__telemetry__bytes__in__place(' \
  'RankGroupProcessSupervisor26accept__terminal__response(' \
  'PlanFrameBuffer7current(' \
  'ValidatedPlanFrame8copy__to(' \
  'PlanFrameBuffer12encode__plan('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'rank-group warmed allocation symbol is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_|moonbit_add_string|moonbit_array_copy|memcpy|memmove'; then
  printf '%s\n' 'rank-group warmed exchange path allocates or copies managed storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v \
    'Error|RankGroupProcessFailure|RankGroupWireError|WorkerWireError|RankGroupError'; then
  printf '%s\n' 'rank-group warmed exchange path contains non-error heap allocation' >&2
  exit 1
fi
if ! rg -q 'moonbit_make_bytes\(1, 112\)' "$generated_c"; then
  printf '%s\n' 'rank-group allocation positive control is ineffective' >&2
  exit 1
fi

for file in "$package"/*.mbt; do
  if [ "$(wc -l < "$file")" -gt 500 ]; then
    printf '%s exceeds 500 lines\n' "$file" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux rank-group process boundary/allocation gate passed.'
