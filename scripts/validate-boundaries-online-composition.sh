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

# The transport-neutral coordinator is the sole composition boundary for one
# preparation pool, one persistent online instance, and one framed-event
# workspace. Its public values are opaque capabilities; none may reveal the
# mutable inner owners, generation scalars, or a parallel receipt authority.
if [ ! -f service/online_session/pkg.generated.mbti ]; then
  printf '%s\n' 'online session generated interface is required' >&2
  failed=1
else
  expected_online_coordinator_surface="$(cat <<'EOF'
pub fn LunaOnlineFramedCoordinator::begin_drain(Self) -> Unit
pub fn LunaOnlineFramedCoordinator::copy_framed_event_chunk_to(Self, FixedArray[Byte], destination_offset~ : Int, maximum_length~ : Int) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::disconnect(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::framed_event_remaining(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::is_closed(Self) -> Bool
pub fn LunaOnlineFramedCoordinator::maintenance_wait_remaining_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::offer_luna_framed(Self, FixedArray[Byte], source_offset~ : Int, length~ : Int) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::progress(Self) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::progress_off_reactor_maintenance(Self) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::request_cancel(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::take_rejection(Self) -> LunaOnlineFramedRejectionCredit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinator::transport_wait_remaining_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedCoordinatorPreparation::maximum_transport_wait_millis(Self) -> Int raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedCoordinatorPreparation::state(Self) -> LunaOnlineInstancePreparationState
pub fn LunaOnlineFramedCoordinatorPreparation::take_cleanup(Self) -> FailedLunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedCoordinatorPreparation::take_ready(Self) -> LunaOnlineFramedCoordinator raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedEventOffer::confirm(Self, length~ : Int) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedEventOffer::length(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedIngress::consumed_bytes(Self) -> Int
pub fn LunaOnlineFramedIngress::has_request_route(Self) -> Bool
pub fn LunaOnlineFramedIngress::kind(Self) -> LunaOnlineFramedIngressKind
pub fn LunaOnlineFramedIngress::request_route(Self) -> LunaOnlineFramedRequestRoute raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedRejectionCredit::ack(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedRejectionCredit::request_route(Self) -> LunaOnlineFramedRequestRoute raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedRejectionCredit::rule(Self) -> LunaOnlineFramedRejectionRule raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedRejectionCredit::sequence(Self) -> UInt64 raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedRequestRoute::same_request(Self, Self) -> Bool
pub fn prepare_owned_luna_online_framed_coordinator_approved(@tokenizer.TokenizerSpec, @tokenizer.TokenizerDigest, @spec.ModelIdentity, @inference.InferenceLimits, @framed_wire.FramedWireLimits, Int, @request_admission.LunaRequestPreparationStepBudget, @request_admission.LunaRequestPreparationWorkLimit, @request_admission.LunaRequestPreparationStorageBudget, @framed_wire.LunaFramedEventStepBudget, @core.SchedulerBlueprint, @worker_service.WorkerServiceBinding, @worker_executable_file.WorkerExecutableAdmission, @worker_wire.WorkerStartupContract, @worker_wire.EncodedBootstrapSource, @worker_process.WorkerProcessLimits, @approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot, @worker_service.WorkerRestartBackoffPolicy) -> LunaOnlineFramedCoordinatorPreparation raise LunaOnlineInstancePreparationError
EOF
)"
  actual_online_coordinator_surface="$(rg \
    '^pub fn (prepare_owned_luna_online_framed_coordinator|LunaOnlineFramed(Coordinator(Preparation)?|EventOffer|Ingress|RejectionCredit|RequestRoute)::)' \
    service/online_session/pkg.generated.mbti | sort)"
  if [ "$actual_online_coordinator_surface" != \
      "$expected_online_coordinator_surface" ]; then
    printf '%s\n' 'online framed coordinator public method surface drifted' >&2
    failed=1
  fi

  expected_online_coordinator_types="$(cat <<'EOF'
LunaOnlineFramedCoordinator
LunaOnlineFramedCoordinatorError
LunaOnlineFramedCoordinatorPreparation
LunaOnlineFramedCoordinatorProgress
LunaOnlineFramedCoordinatorRule
LunaOnlineFramedEventOffer
LunaOnlineFramedIngress
LunaOnlineFramedIngressKind
LunaOnlineFramedRejectionCredit
LunaOnlineFramedRejectionRule
LunaOnlineFramedRequestRoute
EOF
)"
  actual_online_coordinator_types="$(rg \
    '^pub(\(all\))? (struct|enum|suberror) LunaOnlineFramed(Coordinator|CoordinatorError|CoordinatorPreparation|CoordinatorProgress|CoordinatorRule|EventOffer|Ingress|IngressKind|RejectionCredit|RejectionRule|RequestRoute)( |\{|$)' \
    service/online_session/pkg.generated.mbti |
    sed -E 's/^pub(\(all\))? (struct|enum|suberror) ([A-Za-z0-9_]+).*/\3/' |
    sort)"
  if [ "$actual_online_coordinator_types" != \
      "$expected_online_coordinator_types" ]; then
    printf '%s\n' 'parallel online framed authority type appeared' >&2
    failed=1
  fi

  online_coordinator_private_count="$(rg -c --pcre2 -U \
    'pub struct LunaOnlineFramed(Coordinator|CoordinatorPreparation|EventOffer|Ingress|RejectionCredit|RequestRoute) \{\n  // private fields\n\}' \
    service/online_session/pkg.generated.mbti)"
  if [ "$online_coordinator_private_count" != '6' ] ||
    rg -n --pcre2 -U \
      'pub struct LunaOnlineFramed(Coordinator|CoordinatorPreparation|EventOffer|Ingress|RejectionCredit|RequestRoute) \{(?s:[^}]*)\} derive\([^)]*Debug' \
      service/online_session/pkg.generated.mbti ||
    rg -n --pcre2 \
      '^pub fn LunaOnlineFramed(Coordinator(Preparation)?|EventOffer|Ingress|RejectionCredit|RequestRoute)::(owner|pool|instance|workspace|work|view|ticket|credit|generation|epoch|raw|storage)\(' \
      service/online_session/pkg.generated.mbti ||
    rg -n --pcre2 \
      '^pub fn LunaOnlineFramedRequestRoute::(sequence|generation|epoch|raw|storage)\(' \
      service/online_session/pkg.generated.mbti ||
    rg -n --pcre2 \
      '^pub fn (prepare_owned_luna_online_framed_coordinator|LunaOnlineFramed(Coordinator(Preparation)?|EventOffer|Ingress|RejectionCredit)::).*-> .*(@request_admission\.Luna|@framed_wire\.LunaFramed|LunaOnlineRequestTicket|LunaOnlineEventCredit|LunaOnlineInstance(?:[ ,)\]]|$))' \
      service/online_session/pkg.generated.mbti; then
    printf '%s\n' \
      'online framed coordinator leaked inner owner, storage, or generation authority' >&2
    failed=1
  fi

  expected_online_valtypes="$(cat <<'EOF'
pub(all) enum LunaOnlineFramedIngressKind {
pub struct LunaOnlineFramedIngress {
pub struct LunaOnlineFramedRequestRoute {
pub(all) enum LunaOnlineFramedCoordinatorProgress {
pub(all) enum LunaOnlineFramedRejectionRule {
pub struct LunaOnlineFramedRejectionCredit {
pub struct LunaOnlineFramedEventOffer {
pub struct LunaOnlineFramedSemanticEvent {
pub(all) enum LunaOnlineFramedObservationKind {
pub(all) enum LunaOnlineFramedLatencyKind {
pub struct LunaOnlineFramedObservation {
pub(all) enum LunaOnlineFramedServiceReadiness {
pub(all) enum LunaOnlineFramedOpenKind {
pub struct LunaOnlineFramedOpen {
pub struct LunaOnlineFramedPlanTelemetry {
pub struct LunaOnlineFramedStream {
EOF
)"
  actual_online_valtypes="$(sed -n '/^#valtype$/{n;p;}' \
    service/online_session/coordinator_types.mbt)"
  if [ "$actual_online_valtypes" != "$expected_online_valtypes" ]; then
    printf '%s\n' \
      'online framed scalar offer/ingress/rejection/progress representation drifted' >&2
    failed=1
  fi
  expected_online_observation_kind="$(cat <<'EOF'
pub(all) enum LunaOnlineFramedObservationKind {
  LunaOnlineFramedAdmissionObservation
  LunaOnlineFramedTokenObservation
  LunaOnlineFramedUsageObservation
  LunaOnlineFramedCompletionObservation
  LunaOnlineFramedCancellationObservation
  LunaOnlineFramedDeadlineObservation
  LunaOnlineFramedRequestFailureObservation
  LunaOnlineFramedWorkerFailureObservation
  LunaOnlineFramedWorkerRestartObservation
  LunaOnlineFramedWorkerRequestTerminalObservation
} derive(Eq, @debug.Debug)
EOF
)"
  actual_online_observation_kind="$(sed -n \
    '/^pub(all) enum LunaOnlineFramedObservationKind {/,/^}/p' \
    service/online_session/pkg.generated.mbti)"
  if [ "$actual_online_observation_kind" != \
      "$expected_online_observation_kind" ]; then
    printf '%s\n' 'online framed observation vocabulary drifted' >&2
    failed=1
  fi
  expected_online_coordinator_progress="$(cat <<'EOF'
pub(all) enum LunaOnlineFramedCoordinatorProgress {
  LunaOnlineFramedCoordinatorIdle
  LunaOnlineFramedCoordinatorAwaitingInput
  LunaOnlineFramedCoordinatorAdvanced
  LunaOnlineFramedCoordinatorEventPreparing
  LunaOnlineFramedCoordinatorEventReady
  LunaOnlineFramedCoordinatorSemanticEventReady
  LunaOnlineFramedCoordinatorObservationReady
  LunaOnlineFramedCoordinatorRejectionReady
  LunaOnlineFramedCoordinatorMaintenanceRequired
  LunaOnlineFramedCoordinatorDraining
  LunaOnlineFramedCoordinatorClosed
} derive(Eq, @debug.Debug)
EOF
)"
  actual_online_coordinator_progress="$(sed -n \
    '/^pub(all) enum LunaOnlineFramedCoordinatorProgress {/,/^}/p' \
    service/online_session/pkg.generated.mbti)"
  if [ "$actual_online_coordinator_progress" != \
      "$expected_online_coordinator_progress" ]; then
    printf '%s\n' 'online framed coordinator progress vocabulary drifted' >&2
    failed=1
  fi

  expected_online_service_surface="$(cat <<'EOF'
pub fn LunaOnlineFramedObservation::ack(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::cached_input_tokens(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::input_tokens(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::kind(Self) -> LunaOnlineFramedObservationKind raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::latency_kind(Self) -> LunaOnlineFramedLatencyKind raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::latency_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::output_tokens(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedObservation::total_tokens(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedOpen::kind(Self) -> LunaOnlineFramedOpenKind
pub fn LunaOnlineFramedPlanTelemetry::last_plan_rows(Self) -> Int
pub fn LunaOnlineFramedPlanTelemetry::last_plan_sequence(Self) -> UInt64
pub fn LunaOnlineFramedPlanTelemetry::last_plan_tokens(Self) -> Int
pub fn LunaOnlineFramedSemanticEvent::cancel_and_discard(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedSemanticEvent::delivered(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedSemanticEvent::discard_after_transport_cut(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedSemanticEvent::request_route(Self) -> LunaOnlineFramedRequestRoute raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedSemanticEvent::view(Self) -> @luna_event.LunaEventView raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::begin_drain(Self) -> Unit
pub fn LunaOnlineFramedService::has_luna_graph_runtime_telemetry(Self) -> Bool raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::is_closed(Self) -> Bool
pub fn LunaOnlineFramedService::luna_graph_runtime_telemetry(Self) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::luna_plan_telemetry(Self) -> LunaOnlineFramedPlanTelemetry raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::luna_telemetry(Self) -> LunaOnlineFramedTelemetry raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::maintenance_wait_remaining_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::open_semantic_stream(Self) -> LunaOnlineFramedOpen
pub fn LunaOnlineFramedService::open_stream(Self) -> LunaOnlineFramedOpen
pub fn LunaOnlineFramedService::progress(Self) -> LunaOnlineFramedServiceReadiness raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedService::readiness(Self) -> LunaOnlineFramedServiceReadiness
pub fn LunaOnlineFramedService::take_open_stream(Self, LunaOnlineFramedOpen) -> LunaOnlineFramedStream raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedServicePreparation::framed_limits(Self) -> @framed_wire.FramedWireLimits raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::has_luna_graph_runtime_telemetry(Self) -> Bool raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::inference_limits(Self) -> @inference.InferenceLimits raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::luna_graph_runtime_telemetry(Self) -> @worker_wire.WorkerGraphTelemetry raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::luna_plan_telemetry(Self) -> LunaOnlineFramedPlanTelemetry raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::luna_telemetry(Self) -> LunaOnlineFramedTelemetry raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::maximum_transport_wait_millis(Self) -> Int raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::model_identity(Self) -> @spec.ModelIdentity raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::state(Self) -> LunaOnlineInstancePreparationState
pub fn LunaOnlineFramedServicePreparation::take_cleanup(Self) -> FailedLunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedServicePreparation::take_ready(Self) -> LunaOnlineFramedService raise LunaOnlineInstancePreparationError
pub fn LunaOnlineFramedStream::copy_framed_event_chunk_to(Self, FixedArray[Byte], destination_offset~ : Int, maximum_length~ : Int) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::disconnect(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::framed_event_remaining(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::is_live(Self) -> Bool
pub fn LunaOnlineFramedStream::luna_framed_boundary_clear(Self) -> Bool raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::luna_framed_input_clear(Self) -> Bool raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::luna_framed_receipt_complete(Self) -> Bool raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::maintenance_wait_remaining_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::offer_luna_framed(Self, FixedArray[Byte], source_offset~ : Int, length~ : Int) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::offer_luna_framed_with_receipt(Self, @request_admission.LunaRequestReceipt, FixedArray[Byte], source_offset~ : Int, length~ : Int) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::offer_luna_text_handoff_with_receipt(Self, @request_admission.LunaRequestReceipt, @inference.LunaTextRequestHandoffLease) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::progress(Self) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::request_cancel(Self) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::request_cancel_route(Self, LunaOnlineFramedRequestRoute) -> Unit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::request_route_live(Self, LunaOnlineFramedRequestRoute) -> Bool raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::take_observation(Self) -> LunaOnlineFramedObservation raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::take_rejection(Self) -> LunaOnlineFramedRejectionCredit raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::take_semantic_event(Self) -> LunaOnlineFramedSemanticEvent raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedStream::transport_wait_remaining_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError
pub fn LunaOnlineFramedTelemetry::active_requests(Self) -> Int
pub fn LunaOnlineFramedTelemetry::kv_pages_free(Self) -> Int
pub fn LunaOnlineFramedTelemetry::kv_pages_used(Self) -> Int
pub fn LunaOnlineFramedTelemetry::prefix_entries(Self) -> Int
pub fn LunaOnlineFramedTelemetry::prefix_evictions(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::prefix_hits(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::prefix_lookups(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::prefix_misses(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::prefix_pages(Self) -> Int
pub fn LunaOnlineFramedTelemetry::prefix_publications(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::prefix_tokens_computed(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::prefix_tokens_reused(Self) -> UInt64
pub fn LunaOnlineFramedTelemetry::queue_depth(Self) -> Int
pub fn prepare_owned_luna_online_framed_service_approved(@tokenizer.TokenizerSpec, @tokenizer.TokenizerDigest, @spec.ModelIdentity, @inference.InferenceLimits, @framed_wire.FramedWireLimits, Int, @request_admission.LunaRequestPreparationStepBudget, @request_admission.LunaRequestPreparationWorkLimit, @request_admission.LunaRequestPreparationStorageBudget, @framed_wire.LunaFramedEventStepBudget, @core.SchedulerBlueprint, @worker_service.WorkerServiceBinding, @worker_executable_file.WorkerExecutableAdmission, @worker_wire.WorkerStartupContract, @worker_wire.EncodedBootstrapSource, @worker_process.WorkerProcessLimits, @approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot, @worker_service.WorkerRestartBackoffPolicy) -> LunaOnlineFramedServicePreparation raise LunaOnlineInstancePreparationError
EOF
)"
  actual_online_service_surface="$(rg \
    '^pub fn (prepare_owned_luna_online_framed_service|LunaOnlineFramed(Observation|Open|PlanTelemetry|SemanticEvent|Service|ServicePreparation|Stream|Telemetry)::)' \
    service/online_session/pkg.generated.mbti | sort)"
  if [ "$actual_online_service_surface" != "$expected_online_service_surface" ]; then
    printf '%s\n' 'online framed multi-service public method surface drifted' >&2
    failed=1
  fi

  expected_online_service_types="$(cat <<'EOF'
LunaOnlineFramedLatencyKind
LunaOnlineFramedObservation
LunaOnlineFramedObservationKind
LunaOnlineFramedOpen
LunaOnlineFramedOpenKind
LunaOnlineFramedPlanTelemetry
LunaOnlineFramedSemanticEvent
LunaOnlineFramedService
LunaOnlineFramedServicePreparation
LunaOnlineFramedServiceReadiness
LunaOnlineFramedStream
LunaOnlineFramedTelemetry
EOF
)"
  actual_online_service_types="$(rg \
    '^pub(\(all\))? (struct|enum) LunaOnlineFramed(Observation|ObservationKind|LatencyKind|Open|OpenKind|PlanTelemetry|SemanticEvent|Service|ServicePreparation|ServiceReadiness|Stream|Telemetry)( |\{|$)' \
    service/online_session/pkg.generated.mbti |
    sed -E 's/^pub(\(all\))? (struct|enum) ([A-Za-z0-9_]+).*/\3/' | sort)"
  if [ "$actual_online_service_types" != "$expected_online_service_types" ] ||
    [ "$(rg -c --pcre2 -U \
      'pub struct LunaOnlineFramed(Observation|Open|PlanTelemetry|SemanticEvent|Service|ServicePreparation|Stream|Telemetry) \{\n  // private fields\n\}' \
      service/online_session/pkg.generated.mbti)" != '8' ] ||
    rg -n --pcre2 -U \
      'pub struct LunaOnlineFramed(Observation|Open|PlanTelemetry|SemanticEvent|Service|ServicePreparation|Stream|Telemetry) \{(?s:[^}]*)\} derive\([^)]*Debug' \
      service/online_session/pkg.generated.mbti ||
    rg -n '^pub fn LunaOnlineFramed(Observation|Open|PlanTelemetry|SemanticEvent|Service|ServicePreparation|Stream|Telemetry)::(owner|pool|instance|workspace|work|ticket|credit|epoch|raw|storage)\(' \
      service/online_session/pkg.generated.mbti; then
    printf '%s\n' 'online framed multi-service authority opacity drifted' >&2
    failed=1
  fi
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
