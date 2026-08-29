#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Reuse the exact dual-view payload proof. This gate covers warmed synchronous
# Server ownership, telemetry, and payload helpers only. Async coroutine,
# timer, socket, and worker-process recovery implementations may allocate and
# are deliberately outside this claim.
scripts/validate-luna-online-tcp-buffer-allocations.sh
moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'reusable online TCP Server release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ { candidate = 1; body = $0 ORS; next }
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

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|Bytes5makei|memcpy|memmove|blit'
contains_forbidden_allocation_or_copy() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

positive_body="$(extract_definition 'LunaOnlineTcpOutputScratch3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | rg -q 'moonbit_make_bytes\('; then
  printf '%s\n' 'reusable Server allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'LunaOnlineTcpServer7service(' \
  'LunaOnlineTcpServer6stream(' \
  'LunaOnlineTcpServer14latch__failure(' \
  'LunaOnlineTcpServer11record__log(' \
  'LunaOnlineTcpServer23record__network__accept(' \
  'LunaOnlineTcpServer27record__network__disconnect(' \
  'LunaOnlineTcpServer20record__backpressure(' \
  'LunaOnlineTcpServer19record__observation(' \
  'LunaOnlineTcpServer16refresh__metrics(' \
  'LunaOnlineTcpServer22refresh__metrics__from(' \
  'LunaOnlineTcpServer24release__revoked__output(' \
  'LunaOnlineTcpServer26reset__connection__scalars(' \
  'LunaOnlineTcpServer29begin__connection__retirement(' \
  'LunaOnlineTcpServer32begin__service__drain__if__ready(' \
  'LunaOnlineTcpServer21latch__service__drain(' \
  'LunaOnlineTcpServer24bounded__transport__wait(' \
  'luna__online__tcp__safe__wait(' \
  'luna__online__tcp__rule__code(' \
  'LunaOnlineTcpServer11offer__tail(' \
  'LunaOnlineTcpServer20capture__observation(' \
  'LunaOnlineTcpServer18capture__rejection(' \
  'LunaOnlineTcpServer21map__stream__progress(' \
  'LunaOnlineTcpServer20progress__retirement(' \
  'LunaOnlineTcpServer24progress__service__drain(' \
  'LunaOnlineTcpServer28close__listener__on__reactor(' \
  'LunaOnlineTcpServer12begin__drain(' \
  'LunaOnlineTcpServer15request__cancel(' \
  'LunaOnlineTcpServer17metrics__snapshot(' \
  'LunaOnlineTcpServer13log__snapshot(' \
  'LunaOnlineTcpServer5state(' \
  'LunaOnlineFramedStream8is__live(' \
  'LunaServerTelemetryBridge13valid__sample(' \
  'LunaServerTelemetryBridge15refresh__sample(' \
  'LunaServerTelemetryBridge7refresh(' \
  'LunaServerTelemetryBridge20next__log__timestamp(' \
  'LunaServerTelemetryBridge11record__log(' \
  'LunaServerTelemetryBridge23record__network__accept(' \
  'LunaServerTelemetryBridge27record__network__disconnect(' \
  'LunaServerTelemetryBridge20record__backpressure(' \
  'LunaServerTelemetryBridge26record__network__rejection(' \
  'LunaServerTelemetryBridge32record__network__rejection__only(' \
  'LunaServerTelemetryBridge28record__admission__rejection(' \
  'LunaServerTelemetryBridge19record__observation(' \
  'LunaServerTelemetryBridge27record__observation__sample(' \
  'LunaServerTelemetryBridge29record__latency__sample__code(' \
  'LunaServerTelemetryBridge13close__gauges(' \
  'LunaServerTelemetryBridge17metrics__snapshot(' \
  'LunaServerTelemetryBridge13log__snapshot('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'reusable Server allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation_or_copy; then
    printf 'reusable Server warmed helper allocates or copies: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

# Observation telemetry is staged once before semantic ACK. A failed ACK may
# retry, but cannot duplicate a counter/log mutation.
if ! rg -q --pcre2 -U \
    'if luna_server_observation_requires_record\(self\.observation_recorded\) \{[\s\S]*self\.observation_recorded = true[\s\S]*observation\.ack\(\)[\s\S]*self\.observations\.clear\(\)[\s\S]*self\.observation_recorded = false' \
    service/online_tcp/server_telemetry.mbt; then
  printf '%s\n' 'reusable Server record-before-ACK order drifted' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    '#valtype\s*priv struct LunaServerObservationSample \{\s*kind : Int\s*latency_kind : Int\s*latency_millis : Int\s*input_tokens : Int\s*output_tokens : Int\s*\}' \
    service/online_tcp/server_telemetry_observation.mbt ||
  ! rg -q --pcre2 -U \
    'LunaOnlineFramedWorkerFailureObservation => 8[\s\S]*LunaOnlineFramedWorkerRequestTerminalObservation => 10' \
    service/online_tcp/server_telemetry_observation.mbt ||
  ! rg -q --pcre2 -U \
    '8 => \{[\s\S]*record_counter\(WorkerFailures\)[\s\S]*record_counter\(Failures\)[\s\S]*record_log\(LunaWorkerFailed, LunaLogWorkerExit\)[\s\S]*record_log\(LunaRequestFailed, LunaLogInternal\)[\s\S]*0[\s\S]*10 => 1' \
    service/online_tcp/server_telemetry_observation.mbt ||
  ! rg -q -F \
    'test "worker loss is durable before Usage disconnect and terminal is count-free" {' \
    service/online_tcp/server_telemetry_bridge_wbtest.mbt; then
  printf '%s\n' \
    'worker-loss durable accounting or count-free terminal proof drifted' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'let latency_kind = observation\.latency_kind\(\)[\s\S]*observation\.latency_millis\(\)[\s\S]*self\.record_observation_sample\(\{[\s\S]*latency_kind: luna_server_latency_kind_code\(latency_kind\)[\s\S]*latency_millis: latency' \
    service/online_tcp/server_telemetry_observation.mbt ||
  ! rg -q --pcre2 -U \
    'if duration_millis < 0 \|\| duration_millis > 86400000[\s\S]*if self\.cold_start_recorded[\s\S]*record_histogram\(ColdStartLatencyMillis, duration_millis\)[\s\S]*self\.cold_start_recorded = true' \
    service/online_tcp/server_telemetry_bridge.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaOnlineTcpServer::record_cold_start_latency_millis[\s\S]*if self\.activity\.reactor_active \|\| self\.activity\.maintenance_active[\s\S]*record_cold_start_latency_millis\(duration_millis\)[\s\S]*-1 => raise luna_online_tcp_error\(LunaOnlineTcpTelemetry\)[\s\S]*0 => raise luna_online_tcp_error\(LunaOnlineTcpLifecycle\)' \
    service/online_tcp/server_telemetry.mbt; then
  printf '%s\n' 'server latency scalar or once-only cold-start seam drifted' >&2
  exit 1
fi

# Publication precedes the bounded wait; positive writes confirm exact bytes
# before Flight release. Zero, unknown, failed, or timed-out writes cannot ACK.
if ! rg -q --pcre2 -U \
    'self\.offers\.push\(offer\)[\s\S]*self\.flights\.push\(flight\)[\s\S]*let wait = self\.bounded_transport_wait[\s\S]*if wait <= 0 \{[\s\S]*begin_connection_retirement[\s\S]*let written = @async\.with_timeout_opt' \
    service/online_tcp/server_output.mbt ||
  ! rg -q --pcre2 -U \
    'let progress = self\.offers\[0\]\.confirm\(length=count\)[\s\S]*try! self\.flights\[0\]\.release\(\)[\s\S]*self\.flights\.clear\(\)[\s\S]*self\.offers\.clear\(\)' \
    service/online_tcp/server_output.mbt; then
  printf '%s\n' 'reusable Server write-confirm-release order drifted' >&2
  exit 1
fi

# Recovery may stale the retained Stream. Retirement authenticates before any
# cut, alternates a failed cut with exactly one lower progress turn, and never
# recursively retries disconnect in that progress turn.
if ! rg -q --pcre2 -U \
    'if !self\.stream\(\)\.is_live\(\) \{[\s\S]*self\.reset_connection_scalars\(\)[\s\S]*LUNA_ONLINE_TCP_SERVER_LISTENING' \
    service/online_tcp/server_progress.mbt ||
  ! rg -q --pcre2 -U \
    'if self\.disconnect_pending && self\.disconnect_progress_pending \{[\s\S]*self\.disconnect_progress_pending = false[\s\S]*self\.stream\(\)\.progress\(\)[\s\S]*if self\.disconnect_pending \{[\s\S]*self\.stream\(\)\.disconnect\(\)' \
    service/online_tcp/server_progress.mbt; then
  printf '%s\n' 'reusable Server stale-stream or alternating-cut order drifted' >&2
  exit 1
fi

# Listener-first drain and the exact max-one/pipeline receipt split are source
# invariants. Pipeline mode also retains one scalar completed-receipt marker,
# but publishes a boundary only with no retained transport or coordinator input.
if ! rg -q --pcre2 -U \
    'if self\.drain_requested && self\.close_listener_pending \{[\s\S]*self\.close_listener_on_reactor\(\)[\s\S]*return LunaOnlineTcpServerDrainingProgress' \
    service/online_tcp/server_progress.mbt ||
  ! rg -q --pcre2 -U \
    'let complete = self[\s\S]*\.stream\(\)[\s\S]*\.luna_framed_receipt_complete\(\)[\s\S]*self\.request_receipt_complete = complete && !self\.pipeline_mode' \
    service/online_tcp/server_ingress.mbt ||
  ! rg -q --pcre2 -U \
    'luna_online_tcp_pipeline_pending_after_record[\s\S]*pending == 0x7fffffff[\s\S]*terminal[\s\S]*pending == 0[\s\S]*let pending_after[\s\S]*record_observation\(observation\)[\s\S]*pipeline_receipts_pending = pending_after[\s\S]*pipeline_receipt_observed = true' \
    service/online_tcp/server_telemetry.mbt ||
  ! rg -q --pcre2 -U \
    'let completed_evidence = self\.pipeline_receipt_observed &&[\s\S]*self\.pipeline_receipts_pending == 0[\s\S]*let pipeline_clear_candidate = self\.pipeline_mode[\s\S]*self\.tail_length == 0[\s\S]*!tail_was_backpressured[\s\S]*luna_framed_input_clear\(\)[\s\S]*luna_framed_boundary_clear\(\)[\s\S]*luna_online_tcp_pipeline_boundary_ready[\s\S]*self\.pipeline_receipt_observed = false[\s\S]*self\.pipeline_receipts_pending = 0[\s\S]*return LunaOnlineTcpServerConnectedProgress[\s\S]*if pipeline_clear_candidate &&[\s\S]*coordinator_input_clear &&[\s\S]*!coordinator_boundary_clear[\s\S]*return LunaOnlineTcpServerAdvanced' \
    service/online_tcp/server_progress.mbt ||
  ! rg -q --pcre2 -U \
    'fn luna_online_tcp_pipeline_boundary_ready[\s\S]*tail_length == 0[\s\S]*!tail_backpressured[\s\S]*coordinator_boundary_clear' \
    service/online_tcp/server_progress.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaOnlineFramedStream::luna_framed_boundary_clear[\s\S]*receipt_slot < 0[\s\S]*queue_count == 0[\s\S]*prepared\.is_empty\(\)[\s\S]*tickets\.is_empty\(\)[\s\S]*pool\.free_lane_count\(\) == service\.pool\.lane_count\(\)[\s\S]*event_credits\.is_empty\(\)[\s\S]*maintenance_kind == 0[\s\S]*observation_kind == 0' \
    service/online_session/framed_stream.mbt ||
  ! rg -q --pcre2 -U \
    'self\.request_receipt_complete = false[\s\S]*self\.pipeline_receipt_observed = false' \
    service/online_tcp/server_lifecycle.mbt; then
  printf '%s\n' 'reusable Server drain or pipeline receipt boundary drifted' >&2
  exit 1
fi

if ! rg -q -F \
    'pub async fn bind_luna_online_tcp_pipeline_server(@socket.Addr, @online_session.LunaOnlineFramedServicePreparation, LunaOnlineTcpLimits, log_capacity~ : Int) -> LunaOnlineTcpServer' \
    service/online_tcp/pkg.generated.mbti; then
  printf '%s\n' 'reusable Server pipeline bind surface drifted' >&2
  exit 1
fi

if rg -n \
    'pub (async )?fn LunaOnlineTcpServer::.*-> .*(Tcp|TcpServer|Bytes|FixedArray|LunaOnlineFramed(Service|Stream|Event|Observation|Rejection))' \
    service/online_tcp/server_*.mbt ||
  [ "$(rg -c '\.accept\(\)' service/online_tcp/server_ingress.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.read\(' service/online_tcp/server_ingress.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.write_once\(' service/online_tcp/server_output.mbt)" -ne 1 ]; then
  printf '%s\n' 'reusable Server escaped its serialized bounded authority shell' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux reusable online TCP Server allocation and authority gate passed.'
