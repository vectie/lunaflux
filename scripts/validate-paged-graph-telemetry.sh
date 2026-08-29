#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

moon test engine/device_step \
  --target native --release --deny-warn --warn-list +73
moon info engine/device_step --target native >/dev/null
moon test engine/device_worker \
  --target native --deny-warn --warn-list +73
moon info engine/device_worker --target native >/dev/null
moon test engine/worker_wire engine/worker_process engine/worker_service \
  --target native --release --deny-warn --warn-list +73
moon info engine/worker_wire engine/worker_process engine/worker_service \
  --target native >/dev/null

whitebox_c='_build/native/release/test/engine/device_step/device_step.whitebox_test.c'
interface='engine/device_step/pkg.generated.mbti'
worker_interface='engine/device_worker/pkg.generated.mbti'
if [ ! -f "$whitebox_c" ] || [ ! -f "$interface" ] ||
  [ ! -f "$worker_interface" ]; then
  printf '%s\n' 'paged graph telemetry release evidence is missing' >&2
  exit 1
fi

extract_definition() {
  pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      ($0 ~ /^struct moonbit_result_/ || $0 ~ /^struct _M0TP/ ||
       $0 ~ /^int32_t / || $0 ~ /^uint64_t /) {
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
  ' "$whitebox_c"
}

hot_body=''
for symbol in \
  'PagedGraphExecutor25graph__runtime__telemetry(' \
  'PagedGraphTelemetryOwner8snapshot(' \
  'PagedGraphTelemetryOwner17record__completed('; do
  body=$(extract_definition "$symbol")
  if [ -z "$body" ]; then
    printf 'paged graph telemetry function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_malloc|moonbit_make_|moonbit_add_string'; then
  printf '%s\n' 'paged graph telemetry warmed report path allocates' >&2
  exit 1
fi
if ! rg -q 'moonbit_make_bytes\(23, 71\)' "$whitebox_c"; then
  printf '%s\n' 'paged graph telemetry allocation positive control is ineffective' >&2
  exit 1
fi

extract_definition_from() {
  source_file="$1"
  pattern="$2"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      ($0 ~ /^struct moonbit_result_/ || $0 ~ /^struct _M0TP/ ||
       $0 ~ /^int32_t / || $0 ~ /^uint64_t / || $0 ~ /^void /) {
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
  ' "$source_file"
}

process_c='_build/native/release/test/engine/worker_process/worker_process.whitebox_test.c'
service_c='_build/native/release/test/engine/worker_service/worker_service.whitebox_test.c'
wire_c="$process_c"
cross_process_hot=''
for specification in \
  "$wire_c|decode__worker__graph__telemetry(" \
  "$wire_c|encode__worker__graph__telemetry(" \
  "$wire_c|WorkerGraphTelemetrySlot7current(" \
  "$wire_c|WorkerGraphTelemetrySlot5store(" \
  "$process_c|accept__graph__telemetry__bytes__in__place(" \
  "$service_c|record__child__report("; do
  source_file=${specification%%|*}
  symbol=${specification#*|}
  body=$(extract_definition_from "$source_file" "$symbol")
  if [ -z "$body" ]; then
    printf 'cross-process graph telemetry function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  cross_process_hot="${cross_process_hot}${body}"
done
if printf '%s\n' "$cross_process_hot" |
  rg -q 'moonbit_make_|moonbit_add_string'; then
  printf '%s\n' 'cross-process graph telemetry warmed path allocates' >&2
  exit 1
fi
unexpected_malloc=$(printf '%s\n' "$cross_process_hot" |
  rg 'moonbit_malloc' | rg -v 'Error' || true)
if [ -n "$unexpected_malloc" ]; then
  printf '%s\n' 'cross-process graph telemetry success path allocates' >&2
  printf '%s\n' "$unexpected_malloc" >&2
  exit 1
fi

if [ "$(grep -Fxc 'const WORKER_GRAPH_TELEMETRY_BYTES : Int = 80' engine/worker_wire/graph_telemetry.mbt)" -ne 1 ] ||
  [ "$(grep -Fxc 'const WORKER_GRAPH_TELEMETRY_VERSION : UInt = 1' engine/worker_wire/graph_telemetry.mbt)" -ne 1 ]; then
  printf '%s\n' 'cross-process graph telemetry fixed wire/version drifted' >&2
  exit 1
fi

for hostile in \
  'graph telemetry sidecar rejects corruption and invalid shape' \
  'graph telemetry shape packing preserves every maximum-width field' \
  'completion is unpublished until exact graph sidecar authenticates' \
  'initial graph counters reject overflow and selected-path mismatch' \
  'service graph telemetry rejects regressions and replacement substitution' \
  'service graph telemetry accepts only a genuinely saturated child step' \
  'server metrics consume graph deltas once and reject regression'; do
  if ! rg -F -q "$hostile" engine/worker_wire engine/worker_process \
      engine/worker_service service/online_tcp --glob '*test.mbt'; then
    printf 'cross-process graph telemetry hostile test is missing: %s\n' \
      "$hostile" >&2
    exit 1
  fi
done

if ! rg -F -q 'GraphHits => 20' metrics/instance/vocabulary.mbt ||
  ! rg -F -q 'GraphMisses => 21' metrics/instance/vocabulary.mbt ||
  ! rg -F -q 'self.metrics.record_counter_u64(GraphHits' \
    service/online_tcp/server_telemetry_bridge.mbt ||
  ! rg -F -q 'self.metrics.record_counter_u64(GraphMisses' \
    service/online_tcp/server_telemetry_bridge.mbt; then
  printf '%s\n' 'instance graph hit/miss metrics composition is missing' >&2
  exit 1
fi

for type in PagedGraphRuntimeTelemetry PagedGraphSelectedShapeClass; do
  if ! awk -v type="$type" '
    $0 == "pub struct " type " {" { seen = 1; next }
    seen && $0 == "  // private fields" { private_fields = 1; next }
    seen && $0 == "}" { exit !(private_fields == 1) }
    END { if (!seen) exit 1 }
  ' "$interface"; then
    printf 'paged graph telemetry type is not opaque: %s\n' "$type" >&2
    exit 1
  fi
done

if rg -n 'pub fn (new_paged_graph_telemetry|PagedGraphTelemetryOwner)|Bool.*[Hh]it|[Hh]it.*Bool' \
    engine/device_step/graph_telemetry.mbt "$interface"; then
  printf '%s\n' 'paged graph telemetry exposes caller-asserted outcome authority' >&2
  exit 1
fi

if [ "$(grep -Fxc 'pub fn DeviceWorkerOwner::graph_runtime_telemetry(Self) -> @device_step.PagedGraphRuntimeTelemetry raise DeviceWorkerError' "$worker_interface")" -ne 1 ] ||
  ! rg -F -q 'executor.graph_runtime_telemetry()' \
    engine/device_worker/graph_telemetry.mbt; then
  printf '%s\n' 'device worker graph telemetry forwarding path is missing' >&2
  exit 1
fi

if [ "$(rg -F -c 'graph_policy~,' engine/device_worker/prepare.mbt)" -ne 2 ] ||
  ! rg -F -q 'authorization.graph_metadata().capture_safe()' \
    engine/device_worker/graph_policy.mbt ||
  ! rg -F -q 'LunaValidatedEagerFallback =>' \
    engine/device_worker/graph_policy.mbt; then
  printf '%s\n' \
    'authenticated BF16 production graph-policy composition is missing' >&2
  exit 1
fi

mode_sources=$(rg -F -c \
  'open_paged_ordered_executor(resources).execution_mode()' \
  engine/device_step/paged_executor_greedy_sampling.mbt \
  engine/device_step/i8_executor_prepare.mbt \
  engine/device_step/fp8_executor_prepare.mbt |
  awk -F: '{ count += $2 } END { print count + 0 }')
if [ "$mode_sources" -ne 3 ]; then
  printf 'paged graph telemetry startup-mode derivations drifted: %s\n' \
    "$mode_sources" >&2
  exit 1
fi

wait_line=$(rg -n -F 'ordered.wait_completion()' \
  engine/device_step/paged_executor_run.mbt | cut -d: -f1)
scale_line=$(rg -n -F 'validate_fp8_scale_evidence_v3(self)' \
  engine/device_step/paged_executor_run.mbt | cut -d: -f1)
record_line=$(rg -n -F 'self.graph_telemetry.record_completed()' \
  engine/device_step/paged_executor_run.mbt | cut -d: -f1)
if [ -z "$wait_line" ] || [ -z "$scale_line" ] || [ -z "$record_line" ] ||
  [ "$record_line" -le "$wait_line" ] || [ "$record_line" -le "$scale_line" ]; then
  printf '%s\n' 'graph hit/miss publication is not completion-authenticated' >&2
  exit 1
fi

for hostile in \
  'actual captured mode records only graph hits with authenticated shape' \
  'actual eager mode records misses and never fabricates shape evidence' \
  'graph runtime counters saturate without wrapping or changing selection' \
  'public paged capabilities reject lifecycle and foreign owners' \
  'worker graph telemetry rejects missing executor without fabricating data'; do
  if ! rg -F -q "$hostile" engine/device_step/*test.mbt \
      engine/device_worker/*test.mbt; then
    printf 'paged graph telemetry hostile test is missing: %s\n' "$hostile" >&2
    exit 1
  fi
done

for policy_test in \
  'BF16 production graph policy derives eager and explicit fallback exactly' \
  'BF16 production graph policy does not fabricate capture-required authority' \
  'paged graph policy admits eager fallback only by explicit variant'; do
  if ! rg -F -q "$policy_test" engine/device_worker/*test.mbt \
      engine/device_step/*test.mbt; then
    printf 'BF16 graph policy test is missing: %s\n' "$policy_test" >&2
    exit 1
  fi
done

if ! rg -F -q 'Some(authorization) => Some(authorization.graph_metadata())' \
    engine/device_step/blueprint_full.mbt ||
  ! rg -F -q 'Some(graph) => Some(graph.shape_class())' \
    engine/device_step/blueprint_full.mbt; then
  printf '%s\n' 'selected graph shape is not retained from authorization' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux paged graph runtime telemetry allocation/authority gate passed.'
