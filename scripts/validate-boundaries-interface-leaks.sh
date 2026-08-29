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

# Internal ABI concrete types must never become part of a public package
# interface. Generated interfaces are authoritative for this boundary.
fail_matches \
  'public package interface leaks an internal ABI type:' \
  --glob 'pkg.generated.mbti' --glob '!internal/**' --glob '!tests/**' \
  --glob '!deploy/worker_executable_file/pkg.generated.mbti' \
  'vectie/lunaflux/internal/'

root_owner_calls=$(rg -n '\.root_owner\(' --glob '*.mbt' || true)
if [ "$(printf '%s\n' "$root_owner_calls" | sed '/^$/d' | wc -l | tr -d ' ')" -ne 1 ] ||
  ! printf '%s\n' "$root_owner_calls" |
    rg -q '^engine/worker_process/root_bound_prepare\.mbt:' ||
  printf '%s\n' "$root_owner_calls" |
    rg -v '^engine/worker_process/root_bound_prepare\.mbt:'; then
  printf '%s\n' 'prepared approved-root owner sharing escaped its sole aggregate boundary' >&2
  failed=1
fi

if [ -f engine/rank_group_process/pkg.generated.mbti ] &&
  rg -n 'WorkerApprovedRoots|PreparedWorkerApprovedRoots' \
    engine/rank_group_process/pkg.generated.mbti; then
  printf '%s\n' 'rank-group process leaks approved-root concrete ownership' >&2
  failed=1
fi

fixture_constructor_calls=$(rg -n '@worker_service\.new_fixture\(' \
  --glob '*.mbt' || true)
if [ -n "$fixture_constructor_calls" ] &&
  printf '%s\n' "$fixture_constructor_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/)'; then
  printf '%s\n' 'WorkerService new_fixture escaped fixture/test scope' >&2
  failed=1
fi

online_lease_references=$(rg -n 'OnlineWorkerLease|take_online' \
  --glob '*.mbt' --glob 'pkg.generated.mbti' || true)
if [ -n "$online_lease_references" ] &&
  printf '%s\n' "$online_lease_references" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'online worker lease escaped its owned aggregate/test boundary' >&2
  failed=1
fi

online_transfer_calls=$(rg -n '\.take_online\(' --glob '*.mbt' || true)
if [ -n "$online_transfer_calls" ] &&
  printf '%s\n' "$online_transfer_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'owned online transfer escaped aggregate/test scope' >&2
  failed=1
fi

online_progress_status_calls=$(rg -n \
  '\.progress_status\(|\.progress_terminal_recovery_status\(' \
  --glob '*.mbt' || true)
if [ -n "$online_progress_status_calls" ] &&
  printf '%s\n' "$online_progress_status_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'sanitized online progress status escaped aggregate/test scope' >&2
  failed=1
fi

if ! rg -q '^pub fn OnlineWorkerLease::progress_status\(Self\) -> OnlineWorkerStep raise WorkerServiceError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OnlineWorkerLease::progress_terminal_recovery_status\(Self, @worker_protocol.WorkerFailure\) -> OnlineTerminalRecoveryStatus$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'sanitized online progress status surface drifted' >&2
  failed=1
fi

scheduler_replacement_calls=$(rg -n \
  '\.replace_submitted_completion_with_failure\(' --glob '*.mbt' || true)
if [ -n "$scheduler_replacement_calls" ] &&
  printf '%s\n' "$scheduler_replacement_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^scheduler/core/|^engine/worker_service/)'; then
  printf '%s\n' 'scheduler invalid-completion replacement escaped recovery scope' >&2
  failed=1
fi

owned_online_prepare_calls=$(rg -n 'prepare_owned_online\(|\.take_prepared_online\(|\.commit_prepared_admission\(' \
  --glob '*.mbt' || true)
if [ -n "$owned_online_prepare_calls" ] &&
  printf '%s\n' "$owned_online_prepare_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'prepared online admission escaped aggregate/test scope' >&2
  failed=1
fi

exclusive_admission_references=$(rg -n \
  'PreparedExclusiveAdmission|prepare_exclusive_admission\(|commit_exclusive_admission\(|abort_exclusive_admission\(|has_exclusive_admission\(' \
  --glob '*.mbt' --glob 'pkg.generated.mbti' || true)
if [ -n "$exclusive_admission_references" ] &&
  printf '%s\n' "$exclusive_admission_references" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^scheduler/core/|^engine/worker_service/)'; then
  printf '%s\n' 'exclusive scheduler admission escaped worker-service/test scope' >&2
  failed=1
fi

prepared_clock_references=$(rg -n 'PreparedMonotonicRead' \
  --glob '*.mbt' --glob 'pkg.generated.mbti' || true)
if [ -n "$prepared_clock_references" ] &&
  printf '%s\n' "$prepared_clock_references" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^runtime/monotonic_clock/|^engine/worker_service/)'; then
  printf '%s\n' 'prepared monotonic reader escaped worker-service/test scope' >&2
  failed=1
fi

if ! rg -q '^pub struct PreparedExclusiveAdmission \{$' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -q '^pub fn Scheduler::prepare_exclusive_admission\(Self, TokenizedRequest\) -> PreparedExclusiveAdmission raise SchedulerError$' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -q '^pub fn Scheduler::commit_exclusive_admission\(Self, PreparedExclusiveAdmission, UInt64\) -> ExclusiveAdmissionCommit$' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -q '^pub fn Scheduler::abort_exclusive_admission\(Self, PreparedExclusiveAdmission\) -> Unit$' \
    scheduler/core/pkg.generated.mbti; then
  printf '%s\n' 'exclusive scheduler admission must remain opaque and exact' >&2
  failed=1
fi

raw_transfer_calls=$(rg -n '\.take_raw_ready\(' --glob '*.mbt' || true)
if [ -n "$raw_transfer_calls" ] &&
  printf '%s\n' "$raw_transfer_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/)'; then
  printf '%s\n' 'owned raw transfer escaped fixture/test scope' >&2
  failed=1
fi

if rg -n 'OwnedWorkerServicePreparation::take_ready|WorkerService::prepare_online_claim|OnlineWorkerLease::release' \
  engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'legacy owned/online extraction API remains public' >&2
  failed=1
fi

if [ -f service/framed_wire/pkg.generated.mbti ]; then
  if ! rg -q '^pub fn LunaFramedRequestStepBudget::new\(Int\) -> Self raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::new\(FramedWireLimits, LunaFramedRequestStepBudget\) -> Self$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::required_byte_cells\(FramedWireLimits\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::required_int_cells\(FramedWireLimits\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::begin\(Self\) -> LunaFramedRequestWork raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWork::offer\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> Int raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWork::progress\(Self\) -> LunaFramedRequestProgress raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWork::take_view\(Self\) -> LunaFramedRequestView raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestView::release\(Self\) -> Unit raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti; then
    printf '%s\n' 'Luna framed-request public contract drifted' >&2
    failed=1
  fi
  if [ "$(rg -c '^pub fn LunaFramedRequestStepBudget::' \
      service/framed_wire/pkg.generated.mbti)" != '2' ] ||
    [ "$(rg -c '^pub fn LunaFramedRequestWorkspace::' \
      service/framed_wire/pkg.generated.mbti)" != '4' ] ||
    [ "$(rg -c '^pub fn LunaFramedRequestWork::' \
      service/framed_wire/pkg.generated.mbti)" != '8' ] ||
    [ "$(rg -c '^pub fn LunaFramedRequestView::' \
      service/framed_wire/pkg.generated.mbti)" != '32' ]; then
    printf '%s\n' 'Luna framed-request method set drifted' >&2
    failed=1
  fi
  expected_luna_work_methods="$(printf '%s\n' \
    abort failure last_work_units offer progress state take_view \
    total_work_units | sort)"
  actual_luna_work_methods="$(sed -n \
    's/^pub fn LunaFramedRequestWork::\([^(:]*\).*/\1/p' \
    service/framed_wire/pkg.generated.mbti | sort)"
  if [ "$actual_luna_work_methods" != "$expected_luna_work_methods" ]; then
    printf '%s\n' 'Luna framed-request work authority surface drifted' >&2
    failed=1
  fi
  expected_luna_view_methods="$(printf '%s\n' \
    cache_permission cache_scope_byte_at cache_scope_length \
    content_digest_byte_at context_ceiling deadline_millis has_top_k \
    has_top_p inference_limits input_byte_at input_kind input_length \
    input_token_at length max_new_tokens plan_digest_byte_at \
    protocol_version_wire release request_id_value sampling_mode \
    sampling_seed_value sampling_temperature stop_string_byte_at \
    stop_string_count stop_string_length stop_token_at stop_token_count \
    stream_preference top_k top_p trace_byte_at trace_length | sort)"
  actual_luna_view_methods="$(sed -n \
    's/^pub fn LunaFramedRequestView::\([^(:]*\).*/\1/p' \
    service/framed_wire/pkg.generated.mbti | sort)"
  luna_request_private_count="$(rg -c --pcre2 -U \
    'pub struct LunaFramedRequest(StepBudget|Workspace|Work|View) \{\n  // private fields\n\}' \
    service/framed_wire/pkg.generated.mbti)"
  if [ "$actual_luna_view_methods" != "$expected_luna_view_methods" ] ||
    [ "$luna_request_private_count" != '4' ] ||
    rg -n '^pub fn LunaFramedRequest(StepBudget|Workspace|Work|View)::.*-> .*(FixedArray|Array\[|ArrayView|ReadOnlyArray|Bytes|String|GenerateRequest|ValidatedRequestFrame)' \
      service/framed_wire/pkg.generated.mbti ||
    rg -n '^pub fn LunaFramedRequest(StepBudget|Workspace|Work|View)::(owner|epoch|storage|workspace)\(' \
      service/framed_wire/pkg.generated.mbti ||
    rg -n 'pub struct LunaFramedRequest(StepBudget|Workspace|Work|View).*derive\([^)]*Debug' \
      service/framed_wire/pkg.generated.mbti; then
    printf '%s\n' \
      'Luna framed-request capability opacity or scalar view drifted' >&2
    failed=1
  fi

  if ! rg -q '^pub fn LunaFramedEventStepBudget::new\(Int\) -> Self raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventStepBudget::work_units\(Self\) -> Int$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWorkspace::new\(FramedWireLimits, LunaFramedEventStepBudget\) -> Self raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWorkspace::required_byte_cells\(FramedWireLimits\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWorkspace::required_reference_cells\(FramedWireLimits\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWorkspace::begin\(Self, @luna_event\.LunaEventView\) -> LunaFramedEventWork raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::abort\(Self\) -> Unit raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::failure\(Self\) -> FramedWireError raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::last_work_units\(Self\) -> Int raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::progress\(Self\) -> LunaFramedEventProgress raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::state\(Self\) -> LunaFramedEventWorkState raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::take_view\(Self\) -> LunaFramedEventView raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventWork::total_work_units\(Self\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventView::length\(Self\) -> Int raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventView::copy_chunk_to\(Self, FixedArray\[Byte\], destination_offset~ : Int, source_offset~ : Int, length~ : Int\) -> Int raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedEventView::release\(Self\) -> Unit raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti; then
    printf '%s\n' 'Luna framed-event cooperative public contract drifted' >&2
    failed=1
  fi
  if [ "$(rg -c '^pub fn LunaFramedEventStepBudget::' \
      service/framed_wire/pkg.generated.mbti)" != '2' ] ||
    [ "$(rg -c '^pub fn LunaFramedEventWorkspace::' \
      service/framed_wire/pkg.generated.mbti)" != '4' ] ||
    [ "$(rg -c '^pub fn LunaFramedEventWork::' \
      service/framed_wire/pkg.generated.mbti)" != '7' ] ||
    [ "$(rg -c '^pub fn LunaFramedEventView::' \
      service/framed_wire/pkg.generated.mbti)" != '3' ]; then
    printf '%s\n' 'Luna framed-event capability method set drifted' >&2
    failed=1
  fi
  expected_luna_event_work_methods="$(printf '%s\n' \
    abort failure last_work_units progress state take_view total_work_units | sort)"
  actual_luna_event_work_methods="$(sed -n \
    's/^pub fn LunaFramedEventWork::\([^(:]*\).*/\1/p' \
    service/framed_wire/pkg.generated.mbti | sort)"
  if [ "$actual_luna_event_work_methods" != \
      "$expected_luna_event_work_methods" ]; then
    printf '%s\n' 'Luna framed-event Work authority surface drifted' >&2
    failed=1
  fi
  luna_event_private_count="$(rg -c --pcre2 -U \
    'pub struct LunaFramedEvent(StepBudget|Workspace|Work|View) \{\n  // private fields\n\}' \
    service/framed_wire/pkg.generated.mbti)"
  if [ "$luna_event_private_count" != '4' ] ||
    rg -n '^pub fn LunaFramedEvent(StepBudget|Workspace|Work|View)::.*-> .*(FixedArray|Array\[|ArrayView|ReadOnlyArray|Bytes|String|LunaEventView)' \
      service/framed_wire/pkg.generated.mbti ||
    rg -n '^pub fn LunaFramedEvent(StepBudget|Workspace|Work|View)::(ack|retire|owner|epoch|storage|workspace|event|raw)\(' \
      service/framed_wire/pkg.generated.mbti ||
    rg -n 'pub struct LunaFramedEvent(StepBudget|Workspace|Work|View).*derive\([^)]*Debug' \
      service/framed_wire/pkg.generated.mbti; then
    printf '%s\n' \
      'Luna framed-event capabilities leaked storage, epoch, or ACK authority' >&2
    failed=1
  fi
fi

if ! rg -q '^pub fn OwnedWorkerServicePreparation::take_raw_ready\(Self\) -> WorkerService raise OwnedWorkerServicePreparationError$' \
  engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OwnedWorkerServicePreparation::take_online\(Self, @monotonic_clock.MonotonicClock\) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OwnedWorkerServicePreparation::take_prepared_online\(Self\) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'owned preparation must expose exact raw and online transfers' >&2
  failed=1
fi

if ! rg -q '^pub fn OnlineWorkerLease::retire_terminal_request\(Self\) -> Unit raise WorkerServiceError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OnlineWorkerLease::shutdown_clean_empty\(Self\) -> Unit raise WorkerServiceError$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' \
    'online worker lease must expose exact persistent retire and empty-shutdown seams' >&2
  failed=1
fi

if rg -n 'OnlineWorkerLease::(scheduler|process|handle|request_id|request_generation|publication)' \
  engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'online worker lease exposes raw owner or identity evidence' >&2
  failed=1
fi

expected_online_worker_request_surface="$(cat <<'EOF'
pub fn LunaOnlineWorkerAdmission::kind(Self) -> LunaOnlineWorkerAdmissionKind
pub fn LunaOnlineWorkerAdmission::request(Self) -> LunaOnlineWorkerRequest raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::abort_cancel(Self) -> Unit raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::commit_cancel(Self) -> OnlineCancelReservationCommit raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::expire(Self) -> OnlineExpiry raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::is_live(Self) -> Bool
pub fn LunaOnlineWorkerRequest::matches_publication_route(Self, @core.SchedulerPublicationRoute) -> Bool
pub fn LunaOnlineWorkerRequest::publication_route(Self) -> @core.SchedulerPublicationRoute raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::reserve_cancel(Self) -> OnlineCancelReservationStart raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::retire_terminal(Self) -> Unit raise WorkerServiceError
pub fn LunaOnlineWorkerRequest::take_reserved_token(Self) -> OnlineWorkerPublication raise WorkerServiceError
EOF
)"
actual_online_worker_request_surface="$(rg \
  '^pub fn (LunaOnlineWorkerAdmission|LunaOnlineWorkerRequest)::' \
  engine/worker_service/pkg.generated.mbti | sort)"
expected_online_worker_request_vocabulary="$(cat <<'EOF'
pub(all) enum LunaOnlineWorkerAdmissionKind {
  LunaOnlineWorkerAdmitted
  LunaOnlineWorkerSaturated
  LunaOnlineWorkerDraining
} derive(Eq, @debug.Debug)
pub(all) enum OnlineCancelReservationStart {
  OnlineCancelReserved
  OnlineCancelNaturalTerminal
} derive(Eq, @debug.Debug)
pub(all) enum OnlineCancelReservationCommit {
  OnlineCancelCommitted
  OnlineCancelReservationAbsent
} derive(Eq, @debug.Debug)
EOF
)"
actual_online_worker_request_vocabulary="$(
  sed -n '/^pub(all) enum LunaOnlineWorkerAdmissionKind {/,/^}/p' \
    engine/worker_service/pkg.generated.mbti
  sed -n '/^pub(all) enum OnlineCancelReservationStart {/,/^}/p' \
    engine/worker_service/pkg.generated.mbti
  sed -n '/^pub(all) enum OnlineCancelReservationCommit {/,/^}/p' \
    engine/worker_service/pkg.generated.mbti
)"
if [ "$actual_online_worker_request_surface" != \
    "$expected_online_worker_request_surface" ] ||
  [ "$actual_online_worker_request_vocabulary" != \
    "$expected_online_worker_request_vocabulary" ] ||
  [ "$(rg -c --pcre2 -U \
    'pub struct LunaOnlineWorker(Admission|Request) \{\n  // private fields\n\}' \
    engine/worker_service/pkg.generated.mbti)" != '2' ] ||
  rg -n --pcre2 -U \
    'pub struct LunaOnlineWorker(Admission|Request) \{(?s:[^}]*)\} derive\([^)]*Debug' \
    engine/worker_service/pkg.generated.mbti ||
  rg -n '^pub fn LunaOnlineWorker(Admission|Request)::(owner|lease|scheduler|process|handle|request_id|generation|epoch|raw|storage)\(' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'online worker per-request authority surface drifted' >&2
  failed=1
fi

online_lower_surface="$(rg -n \
  '^pub fn .*-> .*\b(WorkerService|OnlineWorkerLease|AdmittedRequest|ReceivedRequest|TokenizerSpec|LunaPreparedRequestClaim|TokenizedRequest|RequestHandle|SchedulerPublication|IncrementalOutput)\b' \
  service/online_session/pkg.generated.mbti 2>/dev/null || true)"
if [ -n "$online_lower_surface" ]; then
  printf '%s\n' 'online session aggregate must not return its lower owners' >&2
  failed=1
fi

online_wire_surface="$(rg -n \
  'framed_wire|FramedWireLimits|CanonicalEventWriter|EventFrameBuffer|ValidatedEventFrame|LunaOnlineWireFailure' \
  service/online_session --glob '*.mbt' --glob 'moon.pkg' --glob '*.mbti' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' 2>/dev/null || true)"
if [ -n "$online_wire_surface" ] &&
  printf '%s\n' "$online_wire_surface" |
    rg -v 'coordinator_(types|construct|prepare|ingress|progress|events|lifecycle)\.mbt:|moon\.pkg:|pkg\.generated\.mbti:.*prepare_owned_luna_online_framed_(coordinator|service)|pkg\.generated\.mbti:.*prepare_owned_tensor_parallel_luna_online_framed_service|pkg\.generated\.mbti:.*LunaOnlineFramedServicePreparation::framed_limits|pkg\.generated\.mbti:.*"vectie/lunaflux/service/framed_wire",'; then
  printf '%s\n' \
    'base online instance acquired framed-wire state outside its coordinator' >&2
  failed=1
fi

tensor_parallel_framed_definition="$(rg -n \
  '^pub fn prepare_owned_tensor_parallel_luna_online_framed_service\(' \
  service/online_session --glob '*.mbt' 2>/dev/null || true)"
if [ "$(printf '%s\n' "$tensor_parallel_framed_definition" |
  sed '/^$/d' | wc -l | tr -d ' ')" -ne 1 ] ||
  ! printf '%s\n' "$tensor_parallel_framed_definition" |
    rg -q '^service/online_session/coordinator_prepare\.mbt:' ||
  rg -n \
    '@request_admission\.LunaRequestPreparationPool::new|@framed_wire\.LunaFramedEventWorkspace::new|@tokenizer\.TokenizerSpec' \
    service/online_session/prepare_tensor_parallel.mbt; then
  printf '%s\n' \
    'tensor-parallel online preparation escaped its shared coordinator owner' >&2
  failed=1
fi

if ! rg -q \
  '^pub fn prepare_owned_tensor_parallel_luna_online_framed_service\(' \
  service/online_session/pkg.generated.mbti; then
  printf '%s\n' \
    'tensor-parallel framed coordinator surface is missing' >&2
  failed=1
fi

if ! rg -U -q \
  'fn begin_foundation_session[\s\S]*LunaFramedEventAdapter::new[\s\S]*let admission = instance\.begin\(prepared\)' \
  tests/worker_service_e2e/online_session_harness.mbt; then
  printf '%s\n' \
    'online driver must preflight its fallible event adapter before begin' >&2
  failed=1
fi

online_tokenizer_surface="$(rg -n \
  '@request_admission\.(admit|prepare_luna_request)|@tokenizer\.TokenizerSpec|priv tokenizer[[:space:]]*:' \
  service/online_session --glob '*.mbt' --glob 'moon.pkg' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' 2>/dev/null || true)"
if [ -n "$online_tokenizer_surface" ] &&
  printf '%s\n' "$online_tokenizer_surface" |
    rg -v 'coordinator_prepare\.mbt:.*@tokenizer\.TokenizerSpec'; then
  printf '%s\n' \
    'base online instance acquired tokenizer work/state outside coordinator startup' >&2
  failed=1
fi

if rg -q '^  OutputFailure$' service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'online output failure remains a typed aggregate error' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  { ! rg -q '^pub fn prepare_owned_luna_online_instance_approved\(' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn prepare_owned_luna_online_instance_approved\(@tokenizer\.TokenizerDigest, @spec\.ModelIdentity, @inference\.InferenceLimits, @core\.SchedulerBlueprint, @worker_service\.WorkerServiceBinding, @worker_executable_file\.WorkerExecutableAdmission, @worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, @worker_process\.WorkerProcessLimits, @approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRoot, @worker_service\.WorkerRestartBackoffPolicy\) -> LunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError$' service/online_session/pkg.generated.mbti ||
    ! rg -q --pcre2 -U '^pub struct LunaOnlineRequestTicket \{\n  // private fields\n\} derive\(Eq\)$' service/online_session/pkg.generated.mbti ||
    ! rg -q --pcre2 -U '^pub struct LunaOnlineInstanceAdmission \{\n  // private fields\n\}$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstanceAdmission::kind\(Self\) -> LunaOnlineInstanceAdmissionKind$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstanceAdmission::ticket\(Self\) -> LunaOnlineRequestTicket raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::begin\(Self, @request_admission\.LunaPreparedRequest\) -> LunaOnlineInstanceAdmission raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress\(Self, LunaOnlineRequestTicket\) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress_terminalization\(Self, LunaOnlineRequestTicket\) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress_request_retirement\(Self, LunaOnlineRequestTicket\) -> LunaOnlineInstanceRetirementProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::take_event\(Self, LunaOnlineRequestTicket\) -> LunaOnlineEventCredit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineEventCredit::view\(Self\) -> @luna_event\.LunaEventView raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineEventCredit::ack\(Self\) -> Unit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::begin_drain\(Self\) -> Unit$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress_shutdown\(Self\) -> LunaOnlineInstanceShutdownProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::maintenance_wait_remaining_millis\(Self\) -> Int raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::request_cancel\(Self, LunaOnlineRequestTicket\) -> Unit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::check_deadline\(Self, LunaOnlineRequestTicket\) -> Unit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^  LunaOnlineInstanceRequestRetirementRequired$' service/online_session/pkg.generated.mbti; }; then
  printf '%s\n' 'persistent Luna online instance surface drifted' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  { ! rg -q --pcre2 -U \
      'pub struct LunaOnlineEventCredit \{\n  // private fields\n\}' \
      service/online_session/pkg.generated.mbti ||
    rg -n '^pub fn LunaOnlineInstance::(has_event|event_length|copy_event_to|ack_event)\(' \
      service/online_session/pkg.generated.mbti ||
    rg -n '^pub fn LunaOnlineEventCredit::(instance|ticket|event_epoch|epoch|raw)\(' \
      service/online_session/pkg.generated.mbti; }; then
  printf '%s\n' \
    'Luna online event credit must remain opaque with only view/ACK authority' >&2
  failed=1
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
