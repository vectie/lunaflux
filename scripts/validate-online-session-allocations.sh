#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e --target native --release --deny-warn

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
  rg "$forbidden" | rg -v 'Error|Failure' | rg -q .
}

# The aggregate constructor deliberately allocates all owners and fixed
# storage before rooted worker preparation. The same extractor and predicate
# must observe that positive control before checking publication/steady paths.
positive_body="$(extract_definition 'CanonicalEventWriter3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'online-session allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'publish__online__session(' \
  'publish__online__cleanup(' \
  'OnlineSession10has__event(' \
  'OnlineSession13event__length(' \
  'OnlineSession15copy__event__to(' \
  'OnlineSession10ack__event(' \
  'OnlineSession12begin__abort(' \
  'OnlineSession17progress__cleanup(' \
  'OnlineSession8progress(' \
  'OnlineSession15request__cancel(' \
  'OnlineSession15check__deadline(' \
  'OnlineSession18begin__caller__cut(' \
  'OnlineSession20begin__deadline__cut(' \
  'OnlineSession17enforce__deadline(' \
  'OnlineSession23progress__terminal__cut(' \
  'OnlineSession25progress__terminalization(' \
  'OnlineSession28latch__worker__step__failure(' \
  'OnlineSession22prepare__cut__terminal(' \
  'classify__expiry__cut(' \
  'OnlineSession13decode__token(' \
  'OnlineSession21progress__string__cut(' \
  'OnlineSession31prepare__string__stop__terminal(' \
  'OnlineSession22begin__output__failure(' \
  'OnlineSession29consume__natural__publication(' \
  'OnlineSession26prepare__natural__terminal(' \
  'AdmittedRequest25push__token__into__status(' \
  'AdmittedRequest20finish__into__status(' \
  'AdmittedRequest15is__stop__token(' \
  'IncrementalOutput19push__token__status(' \
  'StopPattern7advance(' \
  'IncrementalOutput20finish__into__status(' \
  'incremental__output11copy__range(' \
  'incremental__output21advance__utf8__status(' \
  'TokenizerSpec28copy__decoded__piece__status(' \
  'CanonicalEventWriter29prepare__token__bytes__status(' \
  'CanonicalEventWriter14prepare__usage(' \
  'CanonicalEventWriter18prepare__completed(' \
  'CanonicalEventWriter15prepare__failed(' \
  'OnlineWorkerLease6expire(' \
  'OnlineWorkerLease28progress__terminal__recovery(' \
  'OnlineWorkerLease36progress__terminal__recovery__status(' \
  'OnlineWorkerLease16progress__status(' \
  'classify__online__worker__error(' \
  'WorkerService21recover__flight__impl(' \
  'Scheduler45replace__submitted__completion__with__failure(' \
  'Scheduler41preflight__exhausted__failure__retirement(' \
  'Scheduler38commit__exhausted__failure__retirement(' \
  'Scheduler11build__next(' \
  'OnlineWorkerLease27commit__prepared__admission(' \
  'Scheduler28commit__exclusive__admission(' \
  'PreparedMonotonicRead12read__status(' \
  'now__millis__status(' \
  'CanonicalEventWriter27preflight__online__capacity('; do
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

stop_body="$(extract_definition 'AdmittedRequest15is__stop__token(')"
if [ -z "$stop_body" ] || printf '%s\n' "$stop_body" |
  rg -q 'iterG|ArrayView4iter|FixedArray4iter'; then
  printf '%s\n' 'online stop-token lookup introduced iterator allocation' >&2
  exit 1
fi

# Generated ordering evidence guards exact cancellation, failure latching, and
# the reactor/off-reactor terminalization split.
decode_body="$(extract_definition 'OnlineSession13decode__token(')"
progress_body="$(extract_definition 'OnlineSession8progress(')"
terminal_body="$(extract_definition 'OnlineSession26prepare__natural__terminal(')"
caller_body="$(extract_definition 'OnlineSession18begin__caller__cut(')"
cut_body="$(extract_definition 'OnlineSession23progress__terminal__cut(')"
terminalization_body="$(extract_definition 'OnlineSession25progress__terminalization(')"
output_failure_body="$(extract_definition 'OnlineSession22begin__output__failure(')"
latch_body="$(extract_definition 'OnlineSession28latch__worker__step__failure(')"
step_body="$(extract_definition 'OnlineWorkerLease16progress__status(')"
request_body="$(extract_definition 'OnlineSession15request__cancel(')"
deadline_body="$(extract_definition 'OnlineSession15check__deadline(')"
ack_body="$(extract_definition 'OnlineSession10ack__event(')"
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
if ! printf '%s\n' "$deadline_body" | rg -U -q \
  -- '->\$6\)[\s\S]*return[\s\S]*begin__deadline__cut'; then
  printf '%s\n' 'pinned deadline poll can sample the clock before returning' >&2
  exit 1
fi
if ! printf '%s\n' "$ack_body" | rg -U -q \
  -- 'prepare__failed[\s\S]*->\$7 = 6;[\s\S]*->\$6 = 1;'; then
  printf '%s\n' 'deadline Failed publication is not pinned after Usage acknowledgement' >&2
  exit 1
fi

# Deferred caller intent remains installed until the fallible exact
# reservation/commit returns, and deadline enforcement follows it. The cut
# publishes only after the exact terminal reason is authenticated.
if ! printf '%s\n' "$progress_body" | rg -U -q \
  -- 'begin__caller__cut[\s\S]*->\$14 = 0;[\s\S]*enforce__deadline'; then
  printf '%s\n' 'deferred caller cut is cleared before reservation or after deadline enforcement' >&2
  exit 1
fi
if ! printf '%s\n' "$caller_body" | rg -U -q \
  -- 'reserve__cancel[\s\S]*commit__cancel[\s\S]*->\$13 = 1;'; then
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
if ! printf '%s\n' "$decode_body" | rg -U -q \
  -- 'push__token__into__status[\s\S]*commit__cancel[\s\S]*->\$12 = 1;[\s\S]*prepare__token__bytes__status[\s\S]*->\$7 = 2;[\s\S]*->\$6 = 1;'; then
  printf '%s\n' \
    'string-stop cancellation is not committed before event publication' >&2
  exit 1
fi
if ! printf '%s\n' "$output_failure_body" | rg -U -q \
  -- '->\$18 = 2;[\s\S]*commit__cancel[\s\S]*->\$13 = 3;'; then
  printf '%s\n' 'output-invalid cause is not latched before exact cut publication' >&2
  exit 1
fi
if ! printf '%s\n' "$step_body" | rg -U -q \
  -- 'classify__online__worker__error[\s\S]*->\$5 = 1;[\s\S]*data\.ok ='; then
  printf '%s\n' 'worker step does not retain recovery lifecycle before scalar disposition' >&2
  exit 1
fi
if printf '%s\n' "$progress_body" | rg -q 'progress__terminal__recovery'; then
  printf '%s\n' 'reactor progress calls off-reactor terminal recovery' >&2
  exit 1
fi
if ! printf '%s\n' "$progress_body" | rg -U -q \
  -- 'progress__status[\s\S]*latch__worker__step__failure[\s\S]*data\.ok = 2;'; then
  printf '%s\n' 'reactor progress does not return terminalization-required after failure latch' >&2
  exit 1
fi
if ! printf '%s\n' "$terminalization_body" | rg -U -q \
  -- 'progress__terminal__recovery__status[\s\S]*progress__terminal__cut'; then
  printf '%s\n' 'off-reactor terminalization does not recover before terminal drain' >&2
  exit 1
fi
if ! rg -U -q \
  'OnlineWorkerOutputInvalid => \{\s*if self\.terminal_cut == 0 && self\.string_cut == 0 \{[\s\S]*self\.failure_kind = 2[\s\S]*self\.terminal_cut = 4\s*\}\s*self\.recovery_close = true' \
  service/online_session/termination.mbt ||
  ! rg -U -q \
    'OnlineWorkerUnavailable => \{\s*if self\.terminal_cut == 0 && self\.string_cut == 0 \{[\s\S]*self\.failure_kind = 3[\s\S]*self\.terminal_cut = 4\s*\}\s*self\.recovery_close = true' \
    service/online_session/termination.mbt ||
  ! printf '%s\n' "$latch_body" | rg -U -q \
    -- '->\$18 = 2;[\s\S]*->\$13 = 4;[\s\S]*->\$20 = 1;[\s\S]*return 1;' ||
  ! printf '%s\n' "$latch_body" | rg -U -q \
    -- '->\$18 = 3;[\s\S]*->\$13 = 4;[\s\S]*->\$20 = 1;[\s\S]*return 1;'; then
  printf '%s\n' 'failure latch does not preserve cause and recovery authority' >&2
  exit 1
fi
if ! printf '%s\n' "$terminal_body" | rg -U -q \
  -- '->\$18 = 3;[\s\S]*prepare__usage[\s\S]*->\$7 = 5;'; then
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
  -- 'replace__submitted__completion__with__failure[\s\S]*retire__received[\s\S]*clear__flight[\s\S]*->\$5 = 2;'; then
  printf '%s\n' 'service recovery retires process authority before scheduler failure terminal' >&2
  exit 1
fi

# The exact aggregate/session/cleanup shells must precede rooted authority.
prepare_body="$(extract_definition 'prepare__owned__session(')"
if [ -z "$prepare_body" ]; then
  printf '%s\n' 'online-session owned constructor is missing' >&2
  exit 1
fi
session_line="$(printf '%s\n' "$prepare_body" | rg -n 'OnlineSession\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
cleanup_line="$(printf '%s\n' "$prepare_body" | rg -n 'FailedOnlineSessionPreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
outcome_line="$(printf '%s\n' "$prepare_body" | rg -n 'OnlineSessionPreparation\*\)moonbit_malloc' | head -n 1 | cut -d: -f1)"
rooted_line="$(printf '%s\n' "$prepare_body" | rg -n 'worker__service22prepare__owned__online' | head -n 1 | cut -d: -f1)"
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
suffix="$(printf '%s\n' "$prepare_body" | tail -n "+$(printf '%s\n' "$prepare_body" | rg -n 'take__prepared__online' | head -n 1 | cut -d: -f1)")"
if [ -z "$suffix" ] ||
  printf '%s\n' "$suffix" | contains_forbidden_allocation; then
  printf '%s\n' 'online-session post-transfer publication introduced allocation' >&2
  exit 1
fi

scripts/validate-worker-service-online-lease-allocations.sh
scripts/validate-framed-wire-event-writer-allocations.sh
printf '%s\n' 'LunaFlux online-session allocation gate passed.'
