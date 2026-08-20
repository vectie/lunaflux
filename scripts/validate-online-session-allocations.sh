#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
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
  ' "$generated_c"
}

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string'

contains_forbidden_allocation() {
  # Exclude only MoonBit's typed exception envelope. Error/failure-named
  # production constructors remain visible to the gate.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

contains_warm_begin_allocation() {
  # Rejection branches may allocate MoonBit's typed exception wrapper. Exclude
  # only that generated error-object constructor; a success result or any
  # request-local allocation remains visible to this predicate.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

# The aggregate constructor deliberately allocates all owners and fixed
# storage before rooted worker preparation. The same extractor and predicate
# must observe that positive control before checking publication/steady paths.
positive_body="$(extract_definition 'LunaEventOwner3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'online-session allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'publish__luna__instance(' \
  'publish__luna__cleanup(' \
  'luna__online__admission(' \
  'LunaOnlineInstanceAdmission4kind(' \
  'LunaOnlineInstanceAdmission6ticket(' \
  'LunaOnlineInstance11take__event(' \
  'LunaOnlineEventCredit4view(' \
  'LunaOnlineEventCredit3ack(' \
  'LunaOnlineInstance18ack__event__credit(' \
  'LunaOnlineInstance22mark__event__published(' \
  'LunaOnlineInstance12begin__abort(' \
  'LunaOnlineInstance29progress__request__retirement(' \
  'LunaOnlineInstance14reset__request(' \
  'LunaOnlineInstance22close__terminal__owner(' \
  'LunaOnlineInstance12begin__drain(' \
  'LunaOnlineInstance18progress__shutdown(' \
  'LunaOnlineInstance8progress(' \
  'LunaOnlineInstance15request__cancel(' \
  'LunaOnlineInstance15check__deadline(' \
  'LunaOnlineInstance18begin__caller__cut(' \
  'LunaOnlineInstance20begin__deadline__cut(' \
  'LunaOnlineInstance17enforce__deadline(' \
  'LunaOnlineInstance23progress__terminal__cut(' \
  'LunaOnlineInstance25progress__terminalization(' \
  'LunaOnlineInstance28latch__worker__step__failure(' \
  'LunaOnlineInstance22prepare__cut__terminal(' \
  'classify__expiry__cut(' \
  'LunaOnlineInstance13decode__token(' \
  'LunaOnlineInstance21progress__string__cut(' \
  'LunaOnlineInstance31prepare__string__stop__terminal(' \
  'LunaOnlineInstance22begin__output__failure(' \
  'LunaOnlineInstance29consume__natural__publication(' \
  'LunaOnlineInstance26prepare__natural__terminal(' \
  'LunaPreparedRequestClaim25push__token__into__status(' \
  'LunaPreparedRequestClaim20finish__into__status(' \
  'LunaPreparedRequestClaim15is__stop__token(' \
  'LunaPreparedRequest11take__claim(' \
  'LunaPreparedRequest18stream__preference(' \
  'LunaPreparedRequest15model__identity(' \
  'LunaPreparedRequest17effective__limits(' \
  'IncrementalOutput19push__token__status(' \
  'StopPattern7advance(' \
  'IncrementalOutput20finish__into__status(' \
  'incremental__output11copy__range(' \
  'incremental__output21advance__utf8__status(' \
  'TokenizerSpec28copy__decoded__piece__status(' \
  'LunaEventOwner22publish__token__status(' \
  'LunaEventOwner14publish__usage(' \
  'LunaEventOwner17publish__accepted(' \
  'LunaEventOwner20has__epoch__capacity(' \
  'LunaEventOwner31replace__usage__with__completed(' \
  'LunaEventOwner28replace__usage__with__failed(' \
  'LunaEventOwner6retire(' \
  'LunaEventOwner7discard(' \
  'LunaEventOwner4view(' \
  'OnlineWorkerLease6expire(' \
  'OnlineWorkerLease28progress__terminal__recovery(' \
  'OnlineWorkerLease36progress__terminal__recovery__status(' \
  'OnlineWorkerLease16progress__status(' \
  'OnlineWorkerLease25retire__terminal__request(' \
  'OnlineWorkerLease22shutdown__clean__empty(' \
  'classify__online__worker__error(' \
  'WorkerService21recover__flight__impl(' \
  'Scheduler45replace__submitted__completion__with__failure(' \
  'Scheduler41preflight__exhausted__failure__retirement(' \
  'Scheduler38commit__exhausted__failure__retirement(' \
  'Scheduler11build__next(' \
  'OnlineWorkerLease5admit(' \
  'Scheduler17commit__admission('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'online-session allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'online-session steady/publication path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

# Native release compilation inlines the claim field access and the three
# direct preflight fields below into begin. The generated begin scan therefore
# covers their machine code; these source guards keep that inlining boundary
# scalar and prevent hidden preparation/tokenization work from moving into it.
if ! rg -U -q \
    'pub fn LunaPreparedRequest::input_token_count[^{]*\{\s*self\.require_claim\(\)\.input_tokens\s*\}' \
    service/request_admission/types.mbt ||
  ! rg -U -q \
    'pub fn LunaPreparedRequest::tokenizer_digest[^{]*\{\s*self\.require_claim\(\)\.tokenizer_digest\s*\}' \
    service/request_admission/types.mbt ||
  ! rg -U -q \
    'pub fn LunaPreparedRequest::inference_limits[^{]*\{\s*self\.require_claim\(\)\.inference_limits\s*\}' \
    service/request_admission/types.mbt ||
  ! rg -U -q \
    'pub fn LunaPreparedRequestClaim::scheduler_request[^{]*\{\s*self\.scheduler_request\s*\}' \
    service/request_admission/types.mbt; then
  printf '%s\n' \
    'inlined prepared-request success accessors stopped being scalar field reads' >&2
  exit 1
fi

# Off-reactor preparation owns tokenization and every request-local allocation.
# Warm begin consumes only the prepared capability and must not allocate, call
# the tokenizer, or construct request/output owners. Busy/draining dispositions
# run before its destructive claim.
begin_body="$(extract_definition 'LunaOnlineInstance5begin(')"
if [ -z "$begin_body" ]; then
  printf '%s\n' 'persistent online begin function is missing' >&2
  exit 1
fi
if printf '%s\n' "$begin_body" | contains_warm_begin_allocation; then
  printf '%s\n' 'persistent online begin allocates after off-reactor preparation' >&2
  exit 1
fi
if printf '%s\n' "$begin_body" |
  rg -q 'TokenizerSpec|prepare__luna__request|TokenizedRequest3new|IncrementalOutput3new'; then
  printf '%s\n' 'persistent online begin reintroduced tokenizer work' >&2
  exit 1
fi
if ! rg -U -q \
  'if self\.drain_requested[\s\S]*return luna_online_admission\(LunaOnlineInstanceDraining[\s\S]*if self\.phase != LunaInstanceIdlePhase \{[\s\S]*return luna_online_admission\(LunaOnlineInstanceBusy[\s\S]*has_epoch_capacity\(effective\.max_new_tokens\(\) \+ 3\)[\s\S]*publish_accepted[\s\S]*prepared\.take_claim[\s\S]*\.admit\(claim\.scheduler_request\(\)\)' \
  service/online_session/admission.mbt; then
  printf '%s\n' \
    'persistent online begin drifted from drain/busy/headroom/Accepted/claim/admit order' >&2
  exit 1
fi

stop_body="$(extract_definition 'LunaPreparedRequestClaim15is__stop__token(')"
if [ -z "$stop_body" ] || printf '%s\n' "$stop_body" |
  rg -q 'iterG|ArrayView4iter|FixedArray4iter'; then
  printf '%s\n' 'online stop-token lookup introduced iterator allocation' >&2
  exit 1
fi

# Generated ordering evidence guards exact cancellation, failure latching, and
# the reactor/off-reactor terminalization split.
decode_body="$(extract_definition 'LunaOnlineInstance13decode__token(')"
progress_body="$(extract_definition 'LunaOnlineInstance8progress(')"
terminal_body="$(extract_definition 'LunaOnlineInstance26prepare__natural__terminal(')"
caller_body="$(extract_definition 'LunaOnlineInstance18begin__caller__cut(')"
cut_body="$(extract_definition 'LunaOnlineInstance23progress__terminal__cut(')"
terminalization_body="$(extract_definition 'LunaOnlineInstance25progress__terminalization(')"
output_failure_body="$(extract_definition 'LunaOnlineInstance22begin__output__failure(')"
latch_body="$(extract_definition 'LunaOnlineInstance28latch__worker__step__failure(')"
step_body="$(extract_definition 'OnlineWorkerLease16progress__status(')"
request_body="$(extract_definition 'LunaOnlineInstance15request__cancel(')"
deadline_body="$(extract_definition 'LunaOnlineInstance15check__deadline(')"
ack_body="$(extract_definition 'LunaOnlineInstance18ack__event__credit(')"
if [ -z "$decode_body" ] || [ -z "$progress_body" ] ||
  [ -z "$terminal_body" ] || [ -z "$caller_body" ] || [ -z "$cut_body" ] ||
  [ -z "$terminalization_body" ] || [ -z "$output_failure_body" ] ||
  [ -z "$latch_body" ] || [ -z "$step_body" ] || [ -z "$request_body" ] ||
  [ -z "$deadline_body" ] || [ -z "$ack_body" ]; then
  printf '%s\n' 'online-session failure-order functions are missing' >&2
  exit 1
fi
if ! printf '%s\n' "$progress_body" | rg -U -q \
  -- 'enforce__deadline[\s\S]*has__publication[\s\S]*progress__status'; then
  printf '%s\n' 'automatic deadline enforcement does not precede publication/worker progress' >&2
  exit 1
fi
if ! printf '%s\n' "$request_body" | rg -U -q \
  -- 'enforce__deadline[\s\S]*begin__caller__cut'; then
  printf '%s\n' 'unpinned caller cancellation does not preflight absolute expiry' >&2
  exit 1
fi
if ! rg -U -q \
  'pub fn LunaOnlineInstance::check_deadline[\s\S]*if self\.events\.has_event\(\) \{[\s\S]*return[\s\S]*self\.begin_deadline_cut' \
  service/online_session/termination.mbt; then
  printf '%s\n' 'pinned deadline poll can sample the clock before returning' >&2
  exit 1
fi
if ! rg -U -q \
  'FailureUsageEvent => \{[\s\S]*replace_usage_with_failed[\s\S]*self\.mark_event_published\(FailedEvent\)' \
  service/online_session/owner.mbt; then
  printf '%s\n' 'deadline Failed publication is not pinned after Usage acknowledgement' >&2
  exit 1
fi

# Deferred caller intent remains installed until the fallible exact
# reservation/commit returns, and deadline enforcement follows it. The cut
# publishes only after the exact terminal reason is authenticated.
if ! rg -U -q \
  'if self\.deferred_cancel \{[\s\S]*self\.begin_caller_cut\(lease\)[\s\S]*self\.deferred_cancel = false[\s\S]*if self\.enforce_deadline\(lease\)' \
  service/online_session/progression.mbt; then
  printf '%s\n' 'deferred caller cut is cleared before reservation or after deadline enforcement' >&2
  exit 1
fi
if ! rg -U -q \
  'fn LunaOnlineInstance::begin_caller_cut[\s\S]*reserve_cancel[\s\S]*commit_cancel[\s\S]*self\.terminal_cut = CallerCancellationCut' \
  service/online_session/termination.mbt; then
  printf '%s\n' 'caller cancellation is not bound to an exact committed cut' >&2
  exit 1
fi
if ! printf '%s\n' "$cut_body" | rg -U -q \
  -- 'terminal__reason[\s\S]*prepare__cut__terminal'; then
  printf '%s\n' 'terminal cut publishes before exact terminal authentication' >&2
  exit 1
fi

# The scalar status accessor is inlined by the native compiler, so the exact
# generated transaction is the stronger proof boundary: an authenticated
# match must commit cancellation and publish its private cut state before the
# token writer can expose credit. Writer rejection after that commit must
# select retained cleanup before constructing the public typed failure.
if ! rg -U -q \
  'pub fn IncrementalOutputStatus::stop_matched\([\s\S]*self\.0 == 0 && self\.2' \
  service/incremental_output/types.mbt; then
  printf '%s\n' 'incremental-output matched status is not a scalar accessor' >&2
  exit 1
fi
if ! rg -U -q \
  'fn LunaOnlineInstance::decode_token[\s\S]*push_token_into_status[\s\S]*commit_cancel[\s\S]*self\.string_cut = CancelledStringCut[\s\S]*publish_token_status[\s\S]*self\.mark_event_published\(TokenEvent\)' \
  service/online_session/progression.mbt; then
  printf '%s\n' \
    'string-stop cancellation is not committed before event publication' >&2
  exit 1
fi
if ! rg -U -q \
  'fn LunaOnlineInstance::begin_output_failure[\s\S]*self\.failure_kind = OutputLatchedFailure[\s\S]*commit_cancel[\s\S]*self\.terminal_cut = OutputCancellationCut' \
  service/online_session/progression.mbt; then
  printf '%s\n' 'output-invalid cause is not latched before exact cut publication' >&2
  exit 1
fi
if ! printf '%s\n' "$step_body" | rg -U -q \
  -- 'classify__online__worker__error[\s\S]*->\$[0-9]+ = 1;[\s\S]*data\.ok ='; then
  printf '%s\n' 'worker step does not retain recovery lifecycle before scalar disposition' >&2
  exit 1
fi
if printf '%s\n' "$progress_body" | rg -q 'progress__terminal__recovery'; then
  printf '%s\n' 'reactor progress calls off-reactor terminal recovery' >&2
  exit 1
fi
if ! rg -U -q \
  'let step = try! lease\.progress_status\(\)[\s\S]*if self\.latch_worker_step_failure\(step\) \{[\s\S]*return LunaOnlineInstanceTerminalizationRequired' \
  service/online_session/progression.mbt; then
  printf '%s\n' 'reactor progress does not return terminalization-required after failure latch' >&2
  exit 1
fi
if ! printf '%s\n' "$terminalization_body" | rg -U -q \
  -- 'progress__terminal__recovery__status[\s\S]*progress__terminal__cut'; then
  printf '%s\n' 'off-reactor terminalization does not recover before terminal drain' >&2
  exit 1
fi
if ! rg -U -q \
  'OnlineWorkerOutputInvalid => \{\s*if self\.terminal_cut == NoTerminalCut && self\.string_cut == NoStringCut \{[\s\S]*self\.failure_kind = OutputLatchedFailure[\s\S]*self\.terminal_cut = WorkerFailureCut\s*\}\s*self\.recovery_close = true' \
  service/online_session/termination.mbt ||
  ! rg -U -q \
    'OnlineWorkerUnavailable => \{\s*if self\.terminal_cut == NoTerminalCut && self\.string_cut == NoStringCut \{[\s\S]*self\.failure_kind = WorkerLatchedFailure[\s\S]*self\.terminal_cut = WorkerFailureCut\s*\}\s*self\.recovery_close = true' \
    service/online_session/termination.mbt ||
  ! rg -U -q \
    'OnlineWorkerUnavailable => \{[\s\S]*self\.failure_kind = WorkerLatchedFailure[\s\S]*self\.terminal_cut = WorkerFailureCut[\s\S]*self\.recovery_close = true' \
    service/online_session/termination.mbt; then
  printf '%s\n' 'failure latch does not preserve cause and recovery authority' >&2
  exit 1
fi
if ! rg -U -q \
  'if reason == OnlineWorkerFailed \{[\s\S]*self\.failure_kind = WorkerLatchedFailure[\s\S]*publish_usage[\s\S]*self\.mark_event_published\(FailureUsageEvent\)' \
  service/online_session/progression.mbt; then
  printf '%s\n' 'authenticated worker terminal does not publish Usage-first failure bundle' >&2
  exit 1
fi

replace_body="$(extract_definition 'Scheduler45replace__submitted__completion__with__failure(')"
replacement_preflight_body="$(extract_definition 'Scheduler41preflight__exhausted__failure__retirement(')"
replacement_commit_body="$(extract_definition 'Scheduler38commit__exhausted__failure__retirement(')"
recover_body="$(extract_definition 'WorkerService21recover__flight__impl(')"
if [ -z "$replace_body" ] || [ -z "$replacement_preflight_body" ] ||
  [ -z "$replacement_commit_body" ] || [ -z "$recover_body" ]; then
  printf '%s\n' 'online-session invalid-completion recovery proof functions are missing' >&2
  exit 1
fi
replacement_hot="${replace_body}${replacement_preflight_body}${replacement_commit_body}${recover_body}"
if printf '%s\n' "$replacement_hot" | contains_forbidden_allocation; then
  printf '%s\n' 'invalid-completion terminal replacement allocates on steady path' >&2
  exit 1
fi
if ! printf '%s\n' "$replace_body" | rg -U -q \
  -- 'preflight__exhausted__failure__retirement[\s\S]*commit__exhausted__failure__retirement'; then
  printf '%s\n' 'invalid completion replacement mutates before terminal preflight' >&2
  exit 1
fi
if printf '%s\n' "$replacement_preflight_body" | rg -q 'CompletionBuffer7consume|SchedulePlanBuffer6retire'; then
  printf '%s\n' 'replacement preflight consumes protocol authority' >&2
  exit 1
fi
if ! printf '%s\n' "$replacement_commit_body" | rg -U -q \
  -- 'CompletionBuffer7consume[\s\S]*enqueue__terminal[\s\S]*release__active__request[\s\S]*SchedulePlanBuffer6retire'; then
  printf '%s\n' 'terminal replacement commit order drifted' >&2
  exit 1
fi
if ! printf '%s\n' "$recover_body" | rg -U -q \
  -- 'replace__submitted__completion__with__failure[\s\S]*retire__received[\s\S]*clear__flight[\s\S]*->\$[0-9]+ = 2;'; then
  printf '%s\n' 'service recovery retires process authority before scheduler failure terminal' >&2
  exit 1
fi

# The exact aggregate/session/cleanup shells must precede rooted authority.
prepare_body="$(extract_definition 'prepare__owned__luna__online__instance(')"
if [ -z "$prepare_body" ]; then
  printf '%s\n' 'online-session owned constructor is missing' >&2
  exit 1
fi
session_line="$(printf '%s\n' "$prepare_body" | rg -n 'LunaOnlineInstance\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
cleanup_line="$(printf '%s\n' "$prepare_body" | rg -n 'FailedLunaOnlineInstancePreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
outcome_line="$(printf '%s\n' "$prepare_body" | rg -n 'LunaOnlineInstancePreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
rooted_line="$(printf '%s\n' "$prepare_body" | rg -n 'worker__service14prepare__owned' | head -n 1 | cut -d: -f1)"
deadline_code_line="$(printf '%s\n' "$prepare_body" | rg -n 'deadline__failure__code' | head -n 1 | cut -d: -f1)"
output_code_line="$(printf '%s\n' "$prepare_body" | rg -n 'output__failure__code' | head -n 1 | cut -d: -f1)"
worker_code_line="$(printf '%s\n' "$prepare_body" | rg -n 'worker__failure__code' | head -n 1 | cut -d: -f1)"
if [ -z "$session_line" ] || [ -z "$cleanup_line" ] ||
  [ -z "$outcome_line" ] || [ -z "$rooted_line" ] ||
  [ -z "$deadline_code_line" ] || [ -z "$output_code_line" ] ||
  [ -z "$worker_code_line" ] ||
  [ "$session_line" -ge "$rooted_line" ] ||
  [ "$cleanup_line" -ge "$rooted_line" ] ||
  [ "$outcome_line" -ge "$rooted_line" ] ||
  [ "$deadline_code_line" -ge "$rooted_line" ] ||
  [ "$output_code_line" -ge "$rooted_line" ] ||
  [ "$worker_code_line" -ge "$rooted_line" ]; then
  printf '%s\n' 'online-session owners are not all allocated before rooted preparation' >&2
  exit 1
fi

owned_body="$(extract_definition 'prepare__owned__internal(')"
if [ -z "$owned_body" ]; then
  printf '%s\n' 'owned online internal constructor is missing' >&2
  exit 1
fi
admission_line="$(printf '%s\n' "$owned_body" | rg -n 'Scheduler29prepare__exclusive__admission' | head -n 1 | cut -d: -f1)"
clock_line="$(printf '%s\n' "$owned_body" | rg -n 'MonotonicClock13prepare__read' | head -n 1 | cut -d: -f1)"
lease_line="$(printf '%s\n' "$owned_body" | rg -n 'OnlineWorkerLease\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
owned_outcome_line="$(printf '%s\n' "$owned_body" | rg -n 'OwnedWorkerServicePreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
owned_rooted_line="$(printf '%s\n' "$owned_body" | rg -n 'prepare__exchange__with__approved__roots' | head -n 1 | cut -d: -f1)"
if [ -z "$admission_line" ] || [ -z "$clock_line" ] ||
  [ -z "$lease_line" ] || [ -z "$owned_outcome_line" ] ||
  [ -z "$owned_rooted_line" ] ||
  [ "$admission_line" -ge "$owned_rooted_line" ] ||
  [ "$clock_line" -ge "$owned_rooted_line" ] ||
  [ "$lease_line" -ge "$owned_rooted_line" ] ||
  [ "$owned_outcome_line" -ge "$owned_rooted_line" ]; then
  printf '%s\n' 'online admission/clock/lease shells were not prepared before rooted activation' >&2
  exit 1
fi

exclusive_prepare_body="$(extract_definition 'Scheduler29prepare__exclusive__admission(')"
if [ -z "$exclusive_prepare_body" ] ||
  ! printf '%s\n' "$exclusive_prepare_body" | contains_forbidden_allocation; then
  printf '%s\n' 'exclusive admission pre-root allocation/order control is missing' >&2
  exit 1
fi
if ! printf '%s\n' "$exclusive_prepare_body" | rg -q -- '->\$[0-9]+ = 1;'; then
  printf '%s\n' 'exclusive admission does not publish its reservation after shell allocation' >&2
  exit 1
fi

# After the exact lower take_online succeeds, aggregate publication is only
# owner installation, lease admission, or already-preallocated result return.
suffix="$(printf '%s\n' "$prepare_body" | tail -n "+$(printf '%s\n' "$prepare_body" | rg -n 'take__online' | head -n 1 | cut -d: -f1)")"
if [ -z "$suffix" ] ||
  printf '%s\n' "$suffix" | contains_forbidden_allocation; then
  printf '%s\n' 'online-session post-transfer publication introduced allocation' >&2
  exit 1
fi

scripts/validate-worker-service-online-lease-allocations.sh
scripts/validate-luna-framed-event-allocations.sh
scripts/validate-luna-tokenizer-work-allocations.sh
scripts/validate-framed-request-receipt-allocations.sh
printf '%s\n' 'LunaFlux online-session allocation gate passed.'
