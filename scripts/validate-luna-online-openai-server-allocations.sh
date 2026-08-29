#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Reuse the allocation/ownership proofs for every bounded child engine and for
# the shared TCP scratch/Flight implementation. This script proves the warmed
# synchronous OpenAI composition helpers only; async coroutine, timer, socket,
# and worker-process recovery implementations are deliberately outside it.
service/api_auth/validate-luna-api-auth.sh
service/http1/validate-luna-http1-allocations.sh
service/openai_compat/validate-luna-openai-compat.sh
scripts/validate-luna-online-tcp-server-allocations.sh

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'OpenAI online Server release C output is missing' >&2
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

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|Bytes5makei|moonbit_add_string|memcpy|memmove|blit'
contains_forbidden_allocation_or_copy() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

positive_body="$(extract_definition 'LunaOnlineTcpOutputScratch3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | rg -q 'moonbit_make_bytes\('; then
  printf '%s\n' 'OpenAI Server allocation positive control is ineffective' >&2
  exit 1
fi

# These helpers form the synchronous warmed composition path. Child-work
# allocation proofs are delegated to the three package gates above.
for symbol in \
  'LunaOnlineOpenAIServer7service(' \
  'LunaOnlineOpenAIServer6stream(' \
  'LunaOnlineOpenAIServer14latch__failure(' \
  'LunaOnlineOpenAIServer14clear__receipt(' \
  'LunaOnlineOpenAIServer11clear__http(' \
  'LunaOnlineOpenAIServer13clear__output(' \
  'LunaOnlineOpenAIServer17reset__connection(' \
  'LunaOnlineOpenAIServer11record__log(' \
  'LunaOnlineOpenAIServer23record__network__accept(' \
  'LunaOnlineOpenAIServer27record__network__disconnect(' \
  'LunaOnlineOpenAIServer20record__backpressure(' \
  'LunaOnlineOpenAIServer19record__observation(' \
  'LunaOnlineOpenAIServer16refresh__metrics(' \
  'LunaOnlineOpenAIServer18capture__rejection(' \
  'LunaOnlineOpenAIServer20capture__observation(' \
  'LunaOnlineOpenAIServer24capture__semantic__event(' \
  'LunaOnlineOpenAIServer23map__semantic__progress(' \
  'LunaOnlineOpenAIServer19progress__connected(' \
  'LunaOnlineOpenAIServer20progress__retirement(' \
  'LunaOnlineOpenAIServer24progress__service__drain(' \
  'LunaOnlineOpenAIServer16progress__stream(' \
  'LunaOnlineOpenAIServer16progress__compat(' \
  'LunaOnlineOpenAIServer17progress__inbound(' \
  'LunaOnlineOpenAIServer20begin__success__head(' \
  'LunaOnlineOpenAIServer26enforce__handoff__deadline(' \
  'LunaOnlineOpenAIServer14offer__handoff(' \
  'LunaOnlineOpenAIServer13receipt__wait(' \
  'LunaOnlineOpenAIServer26enforce__receipt__deadline(' \
  'LunaOnlineOpenAIServer18begin__http__error(' \
  'LunaOnlineOpenAIServer18map__http__failure(' \
  'LunaOnlineOpenAIServer17offer__http__tail(' \
  'LunaOnlineOpenAIServer14progress__http(' \
  'LunaOnlineOpenAIServer18progress__response(' \
  'LunaOnlineOpenAIServer12output__wait(' \
  'LunaOnlineOpenAIServer25enforce__output__deadline(' \
  'LunaOnlineOpenAIServer20finish__output__view(' \
  'LunaOnlineOpenAIServer28close__listener__on__reactor(' \
  'LunaOnlineOpenAIServer17begin__retirement(' \
  'LunaOnlineOpenAIServer21latch__service__drain(' \
  'LunaOnlineOpenAIServer32begin__service__drain__if__ready(' \
  'LunaOnlineOpenAIServer12begin__drain(' \
  'LunaOnlineOpenAIServer17metrics__snapshot(' \
  'LunaOnlineOpenAIServer13log__snapshot(' \
  'LunaOnlineTcpOutputWrite20copy__http__response(' \
  'LunaOnlineTcpOutputWrite22copy__openai__response(' \
  'luna__openai__bounded__copy__length(' \
  'luna__openai__handoff__deadline__owner(' \
  'luna__openai__ingress__consumption__valid(' \
  'luna__openai__http__requires__more__input(' \
  'luna__openai__http__resume__phase('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'OpenAI Server allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation_or_copy; then
    printf 'OpenAI Server warmed helper allocates or copies: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

# This scalar predicate is intentionally small enough for the release compiler
# to inline and elide. Pin its allocation-free source body instead of requiring
# an unstable out-of-line C symbol.
if ! rg -q --pcre2 -U \
    'fn luna_openai_terminal_confirmed\([\s\S]*event_terminal && output_length > 0 && output_cursor == output_length' \
    service/online_tcp/openai_server_output.mbt; then
  printf '%s\n' 'OpenAI Server terminal-confirmation predicate drifted' >&2
  exit 1
fi

# Receiving states return to the only transport-read phase after each bounded
# tail/progress quantum. HTTP authentication therefore finishes before body
# storage without a zero-work reactor spin.
if ! rg -q --pcre2 -U \
    'state if luna_openai_http_requires_more_input\(state\) => \{[\s\S]*self\.connection_phase = luna_openai_http_resume_phase\(state\)' \
    service/online_tcp/openai_server_http.mbt ||
  ! rg -q --pcre2 -U \
    'fn luna_openai_http_resume_phase[\s\S]*LUNA_OPENAI_HTTP_RECEIVE[\s\S]*LUNA_OPENAI_HTTP_PROGRESS' \
    service/online_tcp/openai_server_http.mbt; then
  printf '%s\n' 'OpenAI Server HTTP receive-resume order drifted' >&2
  exit 1
fi

# Receipt/Stream deadline authority is authenticated before the typed handoff.
# Only acceptance transfers both authorities; backpressure and drain preserve
# the receipt and handoff lease for deterministic retry or cleanup.
if ! rg -q --pcre2 -U \
    'fn LunaOnlineOpenAIServer::offer_handoff[\s\S]*if !self\.enforce_handoff_deadline\(\)[\s\S]*let handoff = self\.inbound_views\[0\]\.handoff\(\)[\s\S]*offer_luna_text_handoff_with_receipt\(self\.receipts\[0\], handoff\)' \
    service/online_tcp/openai_server_inbound.mbt ||
  ! rg -q --pcre2 -U \
    'let consumed = ingress\.consumed_bytes\(\)[\s\S]*let kind = ingress\.kind\(\)[\s\S]*luna_openai_ingress_consumption_valid[\s\S]*LunaOnlineFramedIngressAccepted => \{[\s\S]*self\.receipts\.clear\(\)[\s\S]*release_after_transfer\(\)[\s\S]*LUNA_OPENAI_STREAM_PROGRESS' \
    service/online_tcp/openai_server_inbound.mbt ||
  ! rg -q --pcre2 -U \
    'LunaOnlineFramedIngressBackpressured => \{[\s\S]*record_backpressure\(\)[\s\S]*LunaOnlineTcpServerBackpressured[\s\S]*LunaOnlineFramedIngressDraining => \{[\s\S]*latch_service_drain\(\)' \
    service/online_tcp/openai_server_inbound.mbt; then
  printf '%s\n' 'OpenAI Server typed-handoff deadline or transfer order drifted' >&2
  exit 1
fi

# The 200 event-stream head has exactly one construction site. It is reached
# only after a semantic event has yielded a compat View; no accept, HTTP, auth,
# inbound, or framed-ingress transition can publish a success response.
if [ "$(rg -n 'self\.begin_success_head\(\)' \
    service/online_tcp/openai_server_*.mbt | awk 'END { print NR }')" -ne 1 ] ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineOpenAIServer::capture_semantic_event[\s\S]*self\.stream\(\)\.take_semantic_event\(\)[\s\S]*self\.outbound_workspace\.begin\([\s\S]*self\.semantic_events\.push\(semantic\)[\s\S]*self\.outbound_works\.push\(work\)[\s\S]*LUNA_OPENAI_COMPAT_PROGRESS' \
    service/online_tcp/openai_server_stream.mbt ||
  ! rg -q --pcre2 -U \
    'LunaOpenAICompatReady => \{[\s\S]*self\.outbound_works\[0\]\.take_view\(\)[\s\S]*self\.outbound_views\.push\(view\)[\s\S]*if !self\.sse_started \{[\s\S]*self\.begin_success_head\(\)' \
    service/online_tcp/openai_server_stream.mbt; then
  printf '%s\n' 'OpenAI Server success head escaped semantic-event ordering' >&2
  exit 1
fi

# A positive socket write confirms only its exact prefix. The View and semantic
# event remain retained until every byte is confirmed; only then does delivery
# occur, followed by the lower ACK on a later Stream progress transition.
if ! rg -q --pcre2 -U \
    'if self\.output_cursor == self\.output_length \{[\s\S]*return self\.finish_output_view\(\)' \
    service/online_tcp/openai_server_output.mbt ||
  ! rg -q --pcre2 -U \
    'if count <= 0 \|\| count > length[\s\S]*self\.output_cursor \+= count[\s\S]*self\.flights\[0\]\.release\(\)' \
    service/online_tcp/openai_server_output.mbt ||
  ! rg -q --pcre2 -U \
    'self\.outbound_views\[0\]\.release\(\)[\s\S]*self\.semantic_events\[0\]\.delivered\(\)[\s\S]*self\.semantic_events\.clear\(\)[\s\S]*luna_openai_terminal_confirmed' \
    service/online_tcp/openai_server_output.mbt; then
  printf '%s\n' 'OpenAI Server confirm-deliver-ACK order drifted' >&2
  exit 1
fi

# Telemetry mutations precede observation ACK and are latched across retries;
# accept/disconnect/rejection/backpressure and instance lifecycle are finite.
if ! rg -q --pcre2 -U \
    'if luna_server_observation_requires_record\(self\.observation_recorded\) \{[\s\S]*self\.observation_recorded = true[\s\S]*observation\.ack\(\)[\s\S]*self\.observations\.clear\(\)[\s\S]*self\.observation_recorded = false' \
    service/online_tcp/openai_server_telemetry.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaOnlineOpenAIServer::record_cold_start_latency_millis[\s\S]*if self\.activity\.reactor_active \|\| self\.activity\.maintenance_active[\s\S]*record_cold_start_latency_millis\(duration_millis\)[\s\S]*-1 => raise luna_online_tcp_error\(LunaOnlineTcpTelemetry\)[\s\S]*0 => raise luna_online_tcp_error\(LunaOnlineTcpLifecycle\)' \
    service/online_tcp/openai_server_telemetry.mbt ||
  [ "$(rg -c 'record_log\(LunaInstanceReady' service/online_tcp/openai_server_prepare.mbt)" -ne 1 ] ||
  [ "$(rg -c 'record_log\(LunaDrainStarted' service/online_tcp/openai_server_progress.mbt)" -ne 1 ] ||
  [ "$(rg -c 'record_log\(LunaInstanceClosed' service/online_tcp/openai_server_progress.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.accept\(\)' service/online_tcp/openai_server_accept.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.read\(' service/online_tcp/openai_server_http.mbt)" -ne 1 ] ||
  [ "$(rg -c '\.write_once\(' service/online_tcp/openai_server_output.mbt)" -ne 1 ]; then
  printf '%s\n' 'OpenAI Server telemetry or serialized I/O shell drifted' >&2
  exit 1
fi

mbti="service/online_tcp/pkg.generated.mbti"
for type in LunaOnlineOpenAIServer LunaOnlineOpenAIServerPreparation; do
  if ! rg -U -q "pub struct ${type} \{\n  // private fields\n\}" "$mbti" ||
    rg -q "impl Debug for ${type}" "$mbti"; then
    printf 'OpenAI Server authority is not opaque: %s\n' "$type" >&2
    exit 1
  fi
done
expected_server_surface="$(cat <<'EOF'
pub async fn LunaOnlineOpenAIServer::progress_on_reactor(Self) -> LunaOnlineTcpServerProgress
pub fn LunaOnlineOpenAIServer::begin_drain(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::graph_runtime_telemetry(Self) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::health(Self) -> LunaOnlineTcpServerHealth
pub fn LunaOnlineOpenAIServer::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineOpenAIServer::log_snapshot(Self) -> @vectie/lunaflux/logging/instance.LunaInstanceLogSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::metrics_snapshot(Self) -> @vectie/lunaflux/metrics/instance.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::record_cold_start_latency_millis(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::request_cancel(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::state(Self) -> LunaOnlineTcpServerState
EOF
)"
actual_server_surface="$(rg '^pub (async )?fn LunaOnlineOpenAIServer::' \
  "$mbti" | sort)"
if [ "$actual_server_surface" != "$expected_server_surface" ] ||
  [ "$(rg -c '^pub async fn bind_luna_online_openai_server' "$mbti")" -ne 1 ] ||
  [ "$(rg -c '^pub fn prepare_luna_online_openai_server' "$mbti")" -ne 1 ] ||
  rg -q --pcre2 \
    '^pub (async )?fn .*-> .*(Bytes|FixedArray|LunaOnlineFramed(Service|Stream|SemanticEvent|Observation|Rejection)|LunaHttp1(View|Work)|LunaOpenAI(Inbound|Compat)(View|Work))' \
    "$mbti"; then
  printf '%s\n' 'OpenAI Server exact authority surface drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux OpenAI online Server warmed-composition allocation and authority gate passed.'
