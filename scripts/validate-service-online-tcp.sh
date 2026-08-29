#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

fail_matches() {
  description=$1
  shift
  if matches=$(rg -n "$@" 2>/dev/null); then
    printf '%s\n%s\n' "$description" "$matches" >&2
    failed=1
  fi
}

# The private online TCP scratch is the sole service importer and caller of its
# dedicated production native ABI. The repository-wide exact owner set also
# includes the separate read-once promotion-verifier-key descriptor owner. The
# scratch may retain two typed views internally, but its generated service
# surface must remain completely empty.
expected_internal_abi_owners="$(cat <<'EOF'
internal/approved_fs
internal/cuda
internal/inference_credential
internal/inherited_drain
internal/monotonic_clock
internal/nccl
internal/online_tcp_buffer_alias
internal/process
internal/promotion_verifier_key
EOF
)"
actual_internal_abi_owners="$(rg -l 'extern\s+"[cC]"|#external' internal \
  --glob '*.mbt' | sed -E 's#^(internal/[^/]+).*#\1#' | sort -u)"
if [ "$actual_internal_abi_owners" != "$expected_internal_abi_owners" ]; then
  printf '%s\n' 'service boundary requires exactly nine internal ABI owners' >&2
  failed=1
fi

if [ ! -f internal/online_tcp_buffer_alias/pkg.generated.mbti ] ||
  [ ! -f service/online_tcp/pkg.generated.mbti ]; then
  printf '%s\n' 'online TCP alias and service generated interfaces are required' >&2
  failed=1
else
  expected_alias_surface='pub fn retain_bytes_as_fixed_array(Bytes) -> FixedArray[Byte]'
  actual_alias_surface="$(rg '^pub ' \
    internal/online_tcp_buffer_alias/pkg.generated.mbti || true)"
  if [ "$actual_alias_surface" != "$expected_alias_surface" ] ||
    rg -n '^pub (struct|enum|type|trait)' \
      internal/online_tcp_buffer_alias/pkg.generated.mbti; then
    printf '%s\n' 'online TCP alias ABI surface drifted or exposed a type' >&2
    failed=1
  fi
expected_online_tcp_surface="$(cat <<'EOF'
pub async fn LunaOnlineFramedConnectionPool::progress_on_reactor(Self) -> LunaOnlineFramedConnectionPoolProgress
pub async fn LunaOnlineHttpControlServer::progress_on_reactor(Self) -> LunaOnlineHttpControlProgress
pub async fn LunaOnlineOpenAIConnectionPool::progress_on_reactor(Self) -> LunaOnlineOpenAIConnectionPoolProgress
pub async fn LunaOnlineOpenAIServer::progress_on_reactor(Self) -> LunaOnlineTcpServerProgress
pub async fn LunaOnlineTcpEndpoint::progress_on_reactor(Self) -> LunaOnlineTcpProgress
pub async fn LunaOnlineTcpServer::progress_on_reactor(Self) -> LunaOnlineTcpServerProgress
pub async fn bind_luna_online_framed_connection_pool(@socket.Addr, LunaOnlineFramedConnectionPoolPreparation, LunaOnlineTcpLimits, log_capacity~ : Int) -> LunaOnlineFramedConnectionPool
pub async fn bind_luna_online_http_control_server(LunaOnlineHttpControlPreparation, LunaOnlineTcpLimits) -> LunaOnlineHttpControlServer
pub async fn bind_luna_online_openai_connection_pool(@socket.Addr, LunaOnlineOpenAIConnectionPoolPreparation, LunaOnlineTcpLimits, log_capacity~ : Int) -> LunaOnlineOpenAIConnectionPool
pub async fn bind_luna_online_openai_server(@socket.Addr, LunaOnlineOpenAIServerPreparation, LunaOnlineTcpLimits, log_capacity~ : Int) -> LunaOnlineOpenAIServer
pub async fn bind_luna_online_tcp_pipeline_server(@socket.Addr, @online_session.LunaOnlineFramedServicePreparation, LunaOnlineTcpLimits, log_capacity~ : Int) -> LunaOnlineTcpServer
pub async fn bind_luna_online_tcp_server(@socket.Addr, @online_session.LunaOnlineFramedServicePreparation, LunaOnlineTcpLimits, log_capacity~ : Int) -> LunaOnlineTcpServer
pub async fn bind_luna_online_tcp_single_endpoint(@socket.Addr, @online_session.LunaOnlineFramedCoordinatorPreparation, LunaOnlineTcpLimits) -> LunaOnlineTcpEndpoint
pub fn LunaOnlineFramedConnectionPool::active_connection_count(Self) -> Int
pub fn LunaOnlineFramedConnectionPool::begin_drain(Self) -> Unit
pub fn LunaOnlineFramedConnectionPool::delivered_events(Self) -> UInt64
pub fn LunaOnlineFramedConnectionPool::health(Self) -> LunaOnlineTcpServerHealth
pub fn LunaOnlineFramedConnectionPool::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineFramedConnectionPool::metrics_snapshot(Self) -> @vectie/lunaflux/metrics/instance.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineFramedConnectionPool::network_accepts(Self) -> UInt64
pub fn LunaOnlineFramedConnectionPool::network_disconnects(Self) -> UInt64
pub fn LunaOnlineFramedConnectionPool::readiness(Self) -> LunaOnlineFramedConnectionPoolReadiness
pub fn LunaOnlineFramedConnectionPool::record_cold_start_latency_millis(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineFramedConnectionPool::state(Self) -> LunaOnlineFramedConnectionPoolState
pub fn LunaOnlineFramedConnectionPoolLimits::connection_capacity(Self) -> Int
pub fn LunaOnlineFramedConnectionPoolLimits::io_poll_timeout_millis(Self) -> Int
pub fn LunaOnlineFramedConnectionPoolLimits::new(connection_capacity~ : Int, io_poll_timeout_millis~ : Int, request_work_units~ : Int, event_work_units~ : Int) -> Self raise LunaOnlineTcpError
pub fn LunaOnlineHttpControlServer::begin_drain(Self) -> Unit
pub fn LunaOnlineHttpControlServer::health(Self) -> LunaOnlineTcpServerHealth
pub fn LunaOnlineHttpControlServer::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineHttpControlServer::publish_startup_measurement(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineHttpControlServer::readiness(Self) -> LunaOnlineHttpControlReadiness
pub fn LunaOnlineHttpControlServer::state(Self) -> LunaOnlineHttpControlState
pub fn LunaOnlineOpenAIConnectionPool::active_connection_count(Self) -> Int
pub fn LunaOnlineOpenAIConnectionPool::begin_drain(Self) -> Unit
pub fn LunaOnlineOpenAIConnectionPool::health(Self) -> LunaOnlineTcpServerHealth
pub fn LunaOnlineOpenAIConnectionPool::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineOpenAIConnectionPool::metrics_snapshot(Self) -> @vectie/lunaflux/metrics/instance.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIConnectionPool::readiness(Self) -> LunaOnlineOpenAIConnectionPoolReadiness
pub fn LunaOnlineOpenAIConnectionPool::record_cold_start_latency_millis(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIConnectionPool::state(Self) -> LunaOnlineOpenAIConnectionPoolState
pub fn LunaOnlineOpenAIConnectionPoolLimits::connection_capacity(Self) -> Int
pub fn LunaOnlineOpenAIConnectionPoolLimits::new(connection_capacity~ : Int, output_buffer_bytes~ : Int, io_poll_timeout_millis~ : Int) -> Self raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIConnectionPoolLimits::output_buffer_bytes(Self) -> Int
pub fn LunaOnlineOpenAIServer::begin_drain(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::graph_runtime_telemetry(Self) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::health(Self) -> LunaOnlineTcpServerHealth
pub fn LunaOnlineOpenAIServer::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineOpenAIServer::log_snapshot(Self) -> @vectie/lunaflux/logging/instance.LunaInstanceLogSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::metrics_snapshot(Self) -> @vectie/lunaflux/metrics/instance.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::record_cold_start_latency_millis(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::request_cancel(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineOpenAIServer::state(Self) -> LunaOnlineTcpServerState
pub fn LunaOnlineTcpEndpoint::begin_drain(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::disconnect(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::failure_rule(Self) -> LunaOnlineTcpRule raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineTcpEndpoint::progress_off_reactor_maintenance(Self) -> LunaOnlineTcpProgress raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::rejection_rule(Self) -> LunaOnlineTcpRejectionRule raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::rejection_sequence(Self) -> UInt64 raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::request_cancel(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpEndpoint::state(Self) -> LunaOnlineTcpEndpointState
pub fn LunaOnlineTcpLimits::new(read_chunk_bytes~ : Int, write_chunk_bytes~ : Int, accept_timeout_millis~ : Int, input_idle_timeout_millis~ : Int, write_timeout_millis~ : Int, reactor_transition_budget~ : Int) -> Self raise LunaOnlineTcpError
pub fn LunaOnlineTcpLimits::pool_poll_timeout_millis(Self) -> Int
pub fn LunaOnlineTcpLimits::reactor_transition_budget(Self) -> Int
pub fn LunaOnlineTcpLimits::read_chunk_bytes(Self) -> Int
pub fn LunaOnlineTcpLimits::require_fits_transport_wait(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpLimits::write_chunk_bytes(Self) -> Int
pub fn LunaOnlineTcpServer::begin_drain(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpServer::graph_runtime_telemetry(Self) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineTcpError
pub fn LunaOnlineTcpServer::health(Self) -> LunaOnlineTcpServerHealth
pub fn LunaOnlineTcpServer::local_addr(Self) -> @socket.Addr
pub fn LunaOnlineTcpServer::log_snapshot(Self) -> @vectie/lunaflux/logging/instance.LunaInstanceLogSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineTcpServer::metrics_snapshot(Self) -> @vectie/lunaflux/metrics/instance.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError
pub fn LunaOnlineTcpServer::record_cold_start_latency_millis(Self, Int) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpServer::request_cancel(Self) -> Unit raise LunaOnlineTcpError
pub fn LunaOnlineTcpServer::state(Self) -> LunaOnlineTcpServerState
pub fn prepare_luna_online_framed_connection_pool(@online_session.LunaOnlineFramedServicePreparation, LunaOnlineTcpLimits, LunaOnlineFramedConnectionPoolLimits, @inference.DeadlineBudget) -> LunaOnlineFramedConnectionPoolPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_http_control_for_native(LunaOnlineTcpServer, @http1.LunaHttp1ControlLimits) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_http_control_for_native_pool(LunaOnlineFramedConnectionPool, @http1.LunaHttp1ControlLimits) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_http_control_for_openai_pool_production(LunaOnlineOpenAIConnectionPool, @http1.LunaHttp1ControlLimits) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_http_control_for_openai_pool_qualification(LunaOnlineOpenAIConnectionPool, @http1.LunaHttp1ControlLimits) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_http_control_for_openai_production(LunaOnlineOpenAIServer, @http1.LunaHttp1ControlLimits) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_http_control_for_openai_qualification(LunaOnlineOpenAIServer, @http1.LunaHttp1ControlLimits) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_openai_connection_pool(@online_session.LunaOnlineFramedServicePreparation, LunaOnlineTcpLimits, LunaOnlineOpenAIConnectionPoolLimits, @http1.LunaHttp1Limits, @http1.LunaHttp1StepBudget, @http1.LunaHttp1ResponseStepBudget, @api_auth.LunaApiAuthPolicy, @openai_compat.LunaOpenAICompatStepBudget, @openai_compat.LunaOpenAICompatStepBudget, @openai_compat.LunaOpenAIInboundLimits, @openai_compat.LunaOpenAIChatTemplate, FixedArray[Byte], model_alias_offset~ : Int, model_alias_length~ : Int, FixedArray[Byte], response_id_prefix_offset~ : Int, response_id_prefix_length~ : Int, @inference.CacheScope, max_new_tokens~ : Int, context_ceiling~ : Int, @inference.SamplingParameters, @inference.SamplingSeed, @inference.DeadlineBudget, production_ready? : Bool) -> LunaOnlineOpenAIConnectionPoolPreparation raise LunaOnlineTcpError
pub fn prepare_luna_online_openai_server(@online_session.LunaOnlineFramedServicePreparation, @http1.LunaHttp1Limits, @http1.LunaHttp1StepBudget, @http1.LunaHttp1ResponseStepBudget, @api_auth.LunaApiAuthPolicy, @openai_compat.LunaOpenAICompatStepBudget, @openai_compat.LunaOpenAICompatStepBudget, @openai_compat.LunaOpenAIInboundLimits, @openai_compat.LunaOpenAIChatTemplate, FixedArray[Byte], model_alias_offset~ : Int, model_alias_length~ : Int, FixedArray[Byte], response_id_prefix_offset~ : Int, response_id_prefix_length~ : Int, @inference.CacheScope, max_new_tokens~ : Int, context_ceiling~ : Int, @inference.SamplingParameters, @inference.SamplingSeed, @inference.DeadlineBudget, production_ready? : Bool) -> LunaOnlineOpenAIServerPreparation raise LunaOnlineTcpError
EOF
)"
  actual_online_tcp_surface="$(rg '^pub (async )?fn ' \
    service/online_tcp/pkg.generated.mbti | sort)"
  if [ "$actual_online_tcp_surface" != "$expected_online_tcp_surface" ]; then
    printf '%s\n' 'online TCP endpoint public method surface drifted' >&2
    failed=1
  fi
expected_online_tcp_types="$(cat <<'EOF'
LunaOnlineFramedConnectionPool
LunaOnlineFramedConnectionPoolLimits
LunaOnlineFramedConnectionPoolPreparation
LunaOnlineFramedConnectionPoolProgress
LunaOnlineFramedConnectionPoolReadiness
LunaOnlineFramedConnectionPoolState
LunaOnlineHttpControlPreparation
LunaOnlineHttpControlProgress
LunaOnlineHttpControlReadiness
LunaOnlineHttpControlServer
LunaOnlineHttpControlState
LunaOnlineOpenAIConnectionPool
LunaOnlineOpenAIConnectionPoolLimits
LunaOnlineOpenAIConnectionPoolPreparation
LunaOnlineOpenAIConnectionPoolProgress
LunaOnlineOpenAIConnectionPoolReadiness
LunaOnlineOpenAIConnectionPoolState
LunaOnlineOpenAIServer
LunaOnlineOpenAIServerPreparation
LunaOnlineTcpEndpoint
LunaOnlineTcpEndpointState
LunaOnlineTcpError
LunaOnlineTcpLimits
LunaOnlineTcpProgress
LunaOnlineTcpRejectionRule
LunaOnlineTcpRule
LunaOnlineTcpServer
LunaOnlineTcpServerHealth
LunaOnlineTcpServerProgress
LunaOnlineTcpServerState
EOF
)"
  actual_online_tcp_types="$(rg \
    '^pub(\(all\))? (struct|enum|suberror) LunaOnline(Framed|Http|OpenAI|Tcp)' \
    service/online_tcp/pkg.generated.mbti |
    sed -E 's/^pub(\(all\))? (struct|enum|suberror) ([A-Za-z0-9_]+).*/\3/' |
    sort)"
  if [ "$actual_online_tcp_types" != "$expected_online_tcp_types" ] ||
    [ "$(rg -c --pcre2 -U \
      'pub struct LunaOnline(FramedConnectionPool|FramedConnectionPoolLimits|FramedConnectionPoolPreparation|HttpControlPreparation|HttpControlServer|OpenAIConnectionPool|OpenAIConnectionPoolLimits|OpenAIConnectionPoolPreparation|OpenAIServer|OpenAIServerPreparation|TcpEndpoint|TcpLimits|TcpServer) \{\n  // private fields\n\}' \
      service/online_tcp/pkg.generated.mbti)" -ne 13 ] ||
    rg -n --pcre2 -U \
      'pub struct LunaOnline(FramedConnectionPool|FramedConnectionPoolLimits|FramedConnectionPoolPreparation|HttpControlPreparation|HttpControlServer|OpenAIConnectionPool|OpenAIConnectionPoolLimits|OpenAIConnectionPoolPreparation|OpenAIServer|OpenAIServerPreparation|TcpEndpoint|TcpLimits|TcpServer) \{(?s:[^}]*)\} derive\([^)]*Debug' \
      service/online_tcp/pkg.generated.mbti ||
    rg -n \
      'LunaOnlineTcp(OutputScratch|OutputWrite|OutputFlight)|online_tcp_buffer_alias|vectie/lunaflux/internal/' \
      service/online_tcp/pkg.generated.mbti ||
    rg -n --pcre2 \
      '^pub (async )?fn LunaOnline(Framed|Http|OpenAI|Tcp).*::(owner|server|connection|coordinator|service|stream|slot|route|input|output|offer|flight|sequence|generation|epoch|raw|storage)\(' \
      service/online_tcp/pkg.generated.mbti ||
    rg -n '@socket\.(Tcp|TcpServer)|LunaOnline(Framed|OpenAI)ConnectionSlot' \
      service/online_tcp/pkg.generated.mbti ||
    rg -n --pcre2 \
      '^pub (async )?fn .*-> .*(Bytes|FixedArray|LunaOnlineFramed(Service|Stream|SemanticEvent|Observation|Rejection)|LunaHttp1(View|Work)|LunaOpenAI(Inbound|Compat)(View|Work))' \
      service/online_tcp/pkg.generated.mbti; then
    printf '%s\n' \
      'online TCP endpoint leaked scratch, socket, inner owner, or generation authority' >&2
    failed=1
  fi
  expected_online_tcp_vocabulary="$(cat <<'EOF'
pub(all) enum LunaOnlineFramedConnectionPoolProgress {
  LunaOnlineFramedConnectionAccepted
  LunaOnlineFramedConnectionPoolAdvanced
  LunaOnlineFramedConnectionPoolBackpressured
  LunaOnlineFramedConnectionRetired
  LunaOnlineFramedConnectionPoolDrainingProgress
  LunaOnlineFramedConnectionPoolFailedProgress
  LunaOnlineFramedConnectionPoolClosedProgress
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineFramedConnectionPoolReadiness {
  LunaOnlineFramedConnectionPoolNotReady
  LunaOnlineFramedConnectionPoolReady
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineFramedConnectionPoolState {
  LunaOnlineFramedConnectionPoolListening
  LunaOnlineFramedConnectionPoolActive
  LunaOnlineFramedConnectionPoolDraining
  LunaOnlineFramedConnectionPoolFailed
  LunaOnlineFramedConnectionPoolClosed
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineHttpControlProgress {
  LunaOnlineHttpControlAcceptTimedOut
  LunaOnlineHttpControlConnectedProgress
  LunaOnlineHttpControlAdvanced
  LunaOnlineHttpControlConnectionRetired
  LunaOnlineHttpControlDrainingProgress
  LunaOnlineHttpControlClosedProgress
  LunaOnlineHttpControlFailedProgress
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineHttpControlReadiness {
  LunaOnlineHttpControlNotReady
  LunaOnlineHttpControlReady
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineHttpControlState {
  LunaOnlineHttpControlListening
  LunaOnlineHttpControlConnected
  LunaOnlineHttpControlDraining
  LunaOnlineHttpControlClosed
  LunaOnlineHttpControlFailed
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineOpenAIConnectionPoolProgress {
  LunaOnlineOpenAIConnectionAccepted
  LunaOnlineOpenAIConnectionPoolAdvanced
  LunaOnlineOpenAIConnectionPoolBackpressured
  LunaOnlineOpenAIConnectionRetired
  LunaOnlineOpenAIConnectionPoolDrainingProgress
  LunaOnlineOpenAIConnectionPoolClosedProgress
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineOpenAIConnectionPoolReadiness {
  LunaOnlineOpenAIConnectionPoolNotReady
  LunaOnlineOpenAIConnectionPoolReady
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineOpenAIConnectionPoolState {
  LunaOnlineOpenAIConnectionPoolListening
  LunaOnlineOpenAIConnectionPoolActive
  LunaOnlineOpenAIConnectionPoolDraining
  LunaOnlineOpenAIConnectionPoolClosed
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpRule {
  LunaOnlineTcpLimits
  LunaOnlineTcpPreparation
  LunaOnlineTcpBind
  LunaOnlineTcpAccept
  LunaOnlineTcpRead
  LunaOnlineTcpWrite
  LunaOnlineTcpTimeout
  LunaOnlineTcpCancelled
  LunaOnlineTcpCoordinator
  LunaOnlineTcpTelemetry
  LunaOnlineTcpLifecycle
  LunaOnlineTcpOwnership
} derive(Eq, @debug.Debug)
pub(all) suberror LunaOnlineTcpError {
  LunaOnlineTcpFailed(LunaOnlineTcpRule)
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpEndpointState {
  LunaOnlineTcpListening
  LunaOnlineTcpAccepting
  LunaOnlineTcpConnected
  LunaOnlineTcpDraining
  LunaOnlineTcpCloseRequired
  LunaOnlineTcpClosed
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpProgress {
  LunaOnlineTcpConnectedProgress
  LunaOnlineTcpAdvanced
  LunaOnlineTcpBackpressured
  LunaOnlineTcpMaintenanceRequired
  LunaOnlineTcpRejected
  LunaOnlineTcpCleanupRequired
  LunaOnlineTcpClosedProgress
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpRejectionRule {
  LunaOnlineTcpRejectedFrame
  LunaOnlineTcpRejectedIdentity
  LunaOnlineTcpRejectedDeadline
  LunaOnlineTcpRejectedInput
  LunaOnlineTcpRejectedTokenization
  LunaOnlineTcpRejectedCapacity
  LunaOnlineTcpRejectedService
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpServerProgress {
  LunaOnlineTcpServerListeningProgress
  LunaOnlineTcpServerAcceptTimedOut
  LunaOnlineTcpServerConnectedProgress
  LunaOnlineTcpServerAdvanced
  LunaOnlineTcpServerBackpressured
  LunaOnlineTcpServerObservationRecorded
  LunaOnlineTcpServerConnectionRetired
  LunaOnlineTcpServerDrainingProgress
  LunaOnlineTcpServerClosedProgress
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpServerState {
  LunaOnlineTcpServerListening
  LunaOnlineTcpServerAccepting
  LunaOnlineTcpServerConnected
  LunaOnlineTcpServerRetiring
  LunaOnlineTcpServerDraining
  LunaOnlineTcpServerClosed
} derive(Eq, @debug.Debug)
pub(all) enum LunaOnlineTcpServerHealth {
  LunaOnlineTcpServerHealthy
  LunaOnlineTcpServerFailed
} derive(Eq, @debug.Debug)
EOF
)"
  actual_online_tcp_vocabulary="$(
    sed -n \
      '/^pub(all) enum LunaOnlineFramedConnectionPoolProgress {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n \
      '/^pub(all) enum LunaOnlineFramedConnectionPoolReadiness {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n \
      '/^pub(all) enum LunaOnlineFramedConnectionPoolState {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineHttpControlProgress {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineHttpControlReadiness {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineHttpControlState {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n \
      '/^pub(all) enum LunaOnlineOpenAIConnectionPoolProgress {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n \
      '/^pub(all) enum LunaOnlineOpenAIConnectionPoolReadiness {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n \
      '/^pub(all) enum LunaOnlineOpenAIConnectionPoolState {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpRule {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) suberror LunaOnlineTcpError {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpEndpointState {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpProgress {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpRejectionRule {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpServerProgress {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpServerState {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
    sed -n '/^pub(all) enum LunaOnlineTcpServerHealth {/,/^} derive/p' \
      service/online_tcp/pkg.generated.mbti
  )"
  if [ "$actual_online_tcp_vocabulary" != \
      "$expected_online_tcp_vocabulary" ]; then
    printf '%s\n' 'online TCP lifecycle/error scalar vocabulary drifted' >&2
    failed=1
  fi
  expected_online_tcp_valtypes="$(cat <<'EOF'
pub(all) enum LunaOnlineFramedConnectionPoolState {
pub(all) enum LunaOnlineFramedConnectionPoolReadiness {
pub(all) enum LunaOnlineFramedConnectionPoolProgress {
pub(all) enum LunaOnlineHttpControlState {
pub(all) enum LunaOnlineHttpControlReadiness {
pub(all) enum LunaOnlineHttpControlProgress {
pub(all) enum LunaOnlineOpenAIConnectionPoolState {
pub(all) enum LunaOnlineOpenAIConnectionPoolReadiness {
pub(all) enum LunaOnlineOpenAIConnectionPoolProgress {
pub(all) enum LunaOnlineTcpEndpointState {
pub(all) enum LunaOnlineTcpProgress {
pub(all) enum LunaOnlineTcpRejectionRule {
pub(all) enum LunaOnlineTcpServerState {
pub(all) enum LunaOnlineTcpServerProgress {
pub(all) enum LunaOnlineTcpServerHealth {
EOF
)"
  actual_online_tcp_valtypes="$(
    sed -n '/^#valtype$/{n;p;}' \
      service/online_tcp/framed_pool_types.mbt
    sed -n \
      '/^#valtype$/{n;/LunaOnlineHttpControlState/p;/LunaOnlineHttpControlReadiness/p;/LunaOnlineHttpControlProgress/p;}' \
      service/online_tcp/control_server_types.mbt
    sed -n '/^#valtype$/{n;p;}' \
      service/online_tcp/openai_pool_types.mbt \
      service/online_tcp/endpoint_types.mbt \
      service/online_tcp/server_types.mbt \
      service/online_tcp/server_health_types.mbt
  )"
  if [ "$actual_online_tcp_valtypes" != "$expected_online_tcp_valtypes" ]; then
    printf '%s\n' 'online TCP scalar state/progress representation drifted' >&2
    failed=1
  fi
fi

if ! rg -F -x -q 'supported_targets = "native"' \
    internal/online_tcp_buffer_alias/moon.pkg ||
  ! rg -F -x -q '  "native-stub": [ "alias.c" ],' \
    internal/online_tcp_buffer_alias/moon.pkg ||
  ! rg -F -x -q \
    '  "stub-cc-flags": "-std=c11 -Wall -Wextra -Werror",' \
    internal/online_tcp_buffer_alias/moon.pkg ||
  [ "$(rg -o '"[A-Za-z0-9_]+\.c"' \
    internal/online_tcp_buffer_alias/moon.pkg | wc -l | tr -d ' ')" -ne 1 ]; then
  printf '%s\n' \
    'online TCP alias package must remain native-only with exactly one stub' >&2
  failed=1
fi

if internal_type_leaks=$(rg -n 'vectie/lunaflux/internal/' \
  --glob 'pkg.generated.mbti' --glob '!internal/**' --glob '!tests/**' \
  --glob '!deploy/worker_executable_file/pkg.generated.mbti' 2>/dev/null); then
  printf '%s\n%s\n' \
    'service boundary found a public internal ABI type leak:' \
    "$internal_type_leaks" >&2
  failed=1
fi

online_alias_importers="$(rg -l \
  '"vectie/lunaflux/internal/online_tcp_buffer_alias"' \
  --glob 'moon.pkg' 2>/dev/null || true)"
if [ "$online_alias_importers" != 'service/online_tcp/moon.pkg' ]; then
  printf '%s\n%s\n' \
    'online TCP alias ABI has an unauthorized service importer:' \
    "$online_alias_importers" >&2
  failed=1
fi

online_alias_calls="$(rg -n \
  '@buffer_alias\.retain_bytes_as_fixed_array\(' --glob '*.mbt' \
  2>/dev/null || true)"
if [ "$(printf '%s\n' "$online_alias_calls" | sed '/^$/d' | wc -l | tr -d ' ')" -ne 1 ] ||
  ! printf '%s\n' "$online_alias_calls" |
    rg -q '^service/online_tcp/scratch\.mbt:' ||
  ! rg -q --pcre2 -U \
    "let immutable = Bytes::makei\\(capacity, _ => b'\\\\x00'\\)\\n  let mutable = @buffer_alias\\.retain_bytes_as_fixed_array\\(immutable\\)" \
    service/online_tcp/scratch.mbt; then
  printf '%s\n' \
    'online TCP alias call escaped its exact dynamic-Bytes constructor' >&2
  failed=1
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
