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

if rg -n 'AdmittedRequest|^pub fn admit\(' \
  service/request_admission/pkg.generated.mbti service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'legacy admitted-request/tokenizer admission API remains public' >&2
  failed=1
fi

if ! rg -q '^pub fn LunaPreparedRequest::take_claim\(Self\) -> LunaPreparedRequestClaim raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  rg -n 'claim_scheduler_request|LunaPreparedRequest::scheduler_request' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'Luna prepared shell must expose only destructive claim transfer' >&2
  failed=1
fi

if rg -n --pcre2 -U \
    'pub struct (LunaPreparedRequest|LunaPreparedRequestClaim) \{\n  (?!// private fields)|pub struct LunaPreparedRequest(?:Claim)? \{(?s:[^}]*)\} derive\([^)]*Debug' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'Luna prepared authority must remain opaque and non-Debug' >&2
  failed=1
fi

if [ "$(rg -c '^pub fn LunaPreparedRequest::' \
    service/request_admission/pkg.generated.mbti)" != '8' ] ||
  ! rg -q '^pub fn LunaPreparedRequest::discard\(Self\) -> Unit raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  rg -n --pcre2 \
    '^pub fn LunaPreparedRequest::(receipt|receipt_at_millis|timestamp|deadline|admission_deadline|scheduler_request|is_stop_token|push_token|push_token_into|push_token_into_status|finish_into|finish_into_status|output_finished|output_stopped)\(' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna prepared shell surface escaped binding preflight/destructive transfer' >&2
  failed=1
fi

if [ "$(rg -c '^pub fn LunaPreparedRequestClaim::' \
    service/request_admission/pkg.generated.mbti)" != '5' ] ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::scheduler_request\(Self\) -> @core\.TokenizedRequest raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::release\(Self\) -> Unit raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::is_stop_token\(Self, Int\) -> Bool$' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::push_token_into_status\(' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::finish_into_status\(' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna prepared claim surface escaped scheduler/output/release ownership' >&2
  failed=1
fi

if rg -n --pcre2 \
    '^pub fn .*@inference\.(?:LunaRequestSemantic(?:Lease|View|Storage|Work|Write)|LunaRequestStopToken(?:View|RetentionSlot)|StopConditions|CachePolicy|CacheScope)|^pub fn LunaPreparedRequestClaim::.*(?:Array\[String\]|ReadOnlyArray\[String\]|StringView|\bepoch\b)' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'request admission leaked semantic retention or claim stop/cache representation authority' >&2
  failed=1
fi

claim_scheduler_consumers="$(rg -l \
  'claim\.scheduler_request\(\)' \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
  --glob '!tests/**' . | sed 's#^\./##' | sort || true)"
expected_claim_scheduler_consumers="$(cat <<'EOF'
service/online_session/admission.mbt
service/online_session/online_multi_admission.mbt
EOF
)"
if [ "$claim_scheduler_consumers" != \
  "$expected_claim_scheduler_consumers" ]; then
  printf '%s\n' \
    'prepared scheduler-request borrow escaped the online admission bridge' >&2
  failed=1
fi

direct_claim_releases="$(rg -n 'claim\.release\(\)' \
  service/online_session --glob '*.mbt' | wc -l | tr -d ' ')"
lifecycle_claim_releases="$(rg -n 'self\.release_request_claim\(\)' \
  service/online_session/lifecycle.mbt | wc -l | tr -d ' ')"
multi_lane_claim_releases="$(rg -n 'lane\.claims\[0\]\.release\(\)' \
  service/online_session/online_multi_lifecycle.mbt | wc -l | tr -d ' ')"
if [ "$direct_claim_releases" != '4' ] ||
  [ "$lifecycle_claim_releases" != '2' ] ||
  [ "$multi_lane_claim_releases" != '1' ] ||
  ! rg -q --pcre2 -U \
    'self\.lease_owner\(\)\.admit\(claim\.scheduler_request\(\)\) catch \{[\s\S]*try! claim\.release\(\)[\s\S]*try! self\.events\.discard\(\)' \
    service/online_session/admission.mbt ||
  ! rg -q --pcre2 -U \
      'fn LunaOnlineInstance::close_terminal_owner[\s\S]*lease\.begin_shutdown_maintenance\(\)[\s\S]*self\.maintenance_state = 3[\s\S]*lease\.progress_maintenance\(\)[\s\S]*LunaWorkerServiceClosed => \(\)[\s\S]*self\.release_request_claim\(\)[\s\S]*self\.lease = None[\s\S]*self\.reset_request\(\)' \
    service/online_session/lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'lease\.retire_terminal_request\(\) catch[\s\S]*self\.release_request_claim\(\)[\s\S]*self\.reset_request\(\)' \
    service/online_session/lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'lease\.try_admit_request\(claim\.scheduler_request\(\)\) catch \{[\s\S]*try! claim\.release\(\)[\s\S]*try! self\.events\.discard\(\)[\s\S]*if lower\.kind\(\) != LunaOnlineWorkerAdmitted \{[\s\S]*try! claim\.release\(\)[\s\S]*try! self\.events\.discard\(\)' \
    service/online_session/online_multi_admission.mbt ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineInstance::release_multi_lane[\s\S]*let request = lane\.worker_requests\[0\][\s\S]*let route = try! request\.publication_route\(\)[\s\S]*try! request\.retire_terminal\(\)[\s\S]*self\.multi\.routes\.clear\(route, lane_index\)[\s\S]*lane\.claims\[0\]\.release\(\)[\s\S]*lane\.worker_requests\.clear\(\)[\s\S]*lane\.claims\.clear\(\)' \
    service/online_session/online_multi_lifecycle.mbt; then
  printf '%s\n' \
    'online claim release must follow lower rejection, terminal close, or healthy retirement exactly once' >&2
  failed=1
fi

if rg -n --pcre2 -U \
    'pub struct (LunaRequestPreparationPool|LunaRequestPreparationAdmission|LunaRequestPreparationWork|LunaRequestPreparationStepBudget|LunaRequestPreparationWorkLimit|LunaRequestPreparationStorageBudget) \{\n  (?!// private fields)' \
    service/request_admission/pkg.generated.mbti ||
  rg -n \
    '^pub fn (LunaPreparedRequest|LunaPreparedRequestClaim|LunaRequestPreparation[^:]*)::.*(LunaTokenBuffer|TokenBuffer|LunaIncrementalOutput(Workspace|Work|Lease)|Array\[Int\]|ArrayView\[Int\]|ReadOnlyArray\[Int\])' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna request-preparation capabilities leaked raw storage or representation' >&2
  failed=1
fi

if ! rg -q '^pub fn LunaRequestPreparationPool::try_submit\(Self, ReceivedRequest\) -> LunaRequestPreparationAdmission raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationPool::try_begin_luna_framed\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> LunaRequestPreparationAdmission raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationPool::progress\(Self\) -> LunaRequestPreparationPoolProgress$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationAdmission::consumed_bytes\(Self\) -> Int$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::offer_luna_framed\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> Int raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::luna_framed_receipt_complete\(Self\) -> Bool raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::luna_framed_receipt_remaining_millis\(Self\) -> Int raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::take_prepared\(Self\) -> LunaPreparedRequest raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::last_work_units\(Self\) -> Int raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::total_work_units\(Self\) -> UInt64 raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'Luna preparation submit/progress/take surface drifted' >&2
  failed=1
fi

expected_pool_progress="$(cat <<'EOF'
pub(all) enum LunaRequestPreparationPoolProgress {
  LunaRequestPreparationPoolIdle
  LunaRequestPreparationPoolAwaitingInput
  LunaRequestPreparationPoolAdvanced
} derive(Eq, @debug.Debug)
EOF
)"
actual_pool_progress="$(sed -n \
  '/^pub(all) enum LunaRequestPreparationPoolProgress {/,/^}/p' \
  service/request_admission/pkg.generated.mbti)"
if [ "$actual_pool_progress" != "$expected_pool_progress" ]; then
  printf '%s\n' 'Luna preparation pool progress vocabulary drifted' >&2
  failed=1
fi

if rg -n \
    '^pub fn .*(@framed_wire\.LunaFramedRequest(Workspace|Work|View)|@inference\.AdmissionDeadline)' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'direct framed preparation leaked scanner, receipt, or deadline authority' >&2
  failed=1
fi

expected_receipt_surface="$(cat <<'EOF'
pub fn LunaRequestPreparationPool::try_begin_luna_framed_with_receipt(Self, LunaRequestReceipt, FixedArray[Byte], source_offset~ : Int, length~ : Int) -> LunaRequestPreparationAdmission raise RequestAdmissionError
pub fn LunaRequestReceipt::abort(Self) -> Unit raise RequestAdmissionError
pub fn LunaRequestReceipt::remaining_millis(Self) -> Int raise RequestAdmissionError
pub fn LunaRequestReceiptWorkspace::begin(Self) -> LunaRequestReceipt raise RequestAdmissionError
pub fn LunaRequestReceiptWorkspace::new(@monotonic_clock.MonotonicClock, @inference.DeadlineBudget) -> Self raise RequestAdmissionError
EOF
)"
actual_receipt_surface="$(rg '^pub fn (LunaRequestReceipt|LunaRequestReceiptWorkspace|LunaRequestPreparationPool::try_begin_luna_framed_with_receipt)' \
  service/request_admission/pkg.generated.mbti | sort)"
if [ "$actual_receipt_surface" != "$expected_receipt_surface" ] ||
  [ "$(rg -c --pcre2 -U \
    'pub struct LunaRequestReceipt(Workspace)? \{\n  // private fields\n\}' \
    service/request_admission/pkg.generated.mbti)" != '2' ] ||
  rg -n --pcre2 -U \
    'pub struct LunaRequestReceipt(Workspace)? \{(?s:[^}]*)\} derive\([^)]*Debug' \
    service/request_admission/pkg.generated.mbti ||
  rg -n '^pub fn LunaRequestReceipt(Workspace)?::(clock|deadline|epoch|raw|storage|workspace)\(' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'trusted request receipt authority surface drifted' >&2
  failed=1
fi

if rg -n \
    '^pub (struct|enum) Luna.*Framed.*(Receipt|Admission)' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'direct framed preparation introduced a parallel receipt/admission authority' >&2
  failed=1
fi

if rg -n \
    'GenerateRequest::new|Input::(from_utf8|from_token_ids)|TextInput::|StopConditions::new|CachePolicy::new|materialize_luna|@utf8\.encode|Bytes::make|Map::|HashMap' \
    service/request_admission/luna_framed_receipt.mbt \
    service/request_admission/pool_framed_progress.mbt \
    service/request_admission/pool_output_progress.mbt; then
  printf '%s\n' \
    'direct framed preparation reintroduced object materialization' >&2
  failed=1
fi

if rg -n 'moonbitlang/async|async fn|socket|listener|worker_(process|service)|device_|approved_fs' \
    service/request_admission/pool*.mbt; then
  printf '%s\n' \
    'Luna preparation pool acquired async/socket/process/device authority' >&2
  failed=1
fi

for required_authority_interface in \
  contracts/inference/pkg.generated.mbti \
  model/spec/pkg.generated.mbti \
  scheduler/core/pkg.generated.mbti; do
  if [ ! -f "$required_authority_interface" ]; then
    printf '%s\n' \
      "required authority interface ${required_authority_interface} is missing" >&2
    failed=1
  fi
done


if [ "$failed" -ne 0 ]; then
  exit 1
fi
