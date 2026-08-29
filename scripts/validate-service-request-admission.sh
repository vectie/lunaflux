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

# Request admission is the synchronous tokenizer-worker bridge. It may bind
# contracts, tokenizer, monotonic time, incremental output, and the scheduler
# request value, but it must not acquire transport/process/device authority or
# become an async listener.
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

if [ -d service/request_admission ]; then
  fail_matches \
    'request admission imports a forbidden engine or transport owner:' \
    --glob 'service/request_admission/moon.pkg' \
    'worker_service|worker_process|internal/process|moonbitlang/async|runtime/approved_fs'
  fail_matches \
    'request admission must remain synchronous and native-ABI free:' \
    --glob 'service/request_admission/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  fail_matches \
    'Luna preparation pool acquired async/socket/process/device authority:' \
    --glob 'service/request_admission/pool*.mbt' \
    'moonbitlang/async|async fn|socket|listener|worker_(process|service)|device_|approved_fs'
  fail_matches \
    'direct framed preparation reintroduced object materialization:' \
    'GenerateRequest::new|Input::(from_utf8|from_token_ids)|TextInput::|StopConditions::new|CachePolicy::new|materialize_luna|@utf8\.encode|Bytes::make|Map::|HashMap' \
    service/request_admission/luna_framed_receipt.mbt \
    service/request_admission/pool_framed_progress.mbt \
    service/request_admission/pool_output_progress.mbt
  if rg -n 'begin_bytes' service/request_admission/pool*.mbt ||
    ! rg -F -q 'begin_luna_input(' service/request_admission/pool.mbt ||
    ! rg -q 'const LUNA_PREPARATION_TEXT_INPUT_COPY' \
      service/request_admission/pool_types.mbt ||
    ! rg -q --pcre2 -U \
      'progress_text_input_copy(?s).*write\.push_byte\(text\.utf8\(\)\[lane\.copy_index\]\)(?s).*lane\.copy_index \+= 1(?s).*return(?s).*write\.finish\(\)' \
      service/request_admission/pool_progress.mbt ||
    ! rg -q 'LunaTokenizerWorker::required_byte_cells' \
      service/request_admission/pool_storage.mbt ||
    ! rg -q --pcre2 -U \
      'let per_lane_byte = checked_add_cells\(\s*checked_add_cells\(\s*checked_add_cells\(tokenizer_byte_cells, output_byte_cells\),\s*semantic_byte_cells,?\s*\),\s*framed_byte_cells,?\s*\)' \
      service/request_admission/pool_storage.mbt; then
    printf '%s\n' \
      'Luna preparation pool lost bounded tokenizer input or exact byte accounting' >&2
    failed=1
  fi
  if [ -f service/request_admission/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn prepare_luna_request\(ReceivedRequest, @tokenizer\.TokenizerSpec, @tokenizer\.TokenizerDigest, @spec\.ModelIdentity, @inference\.InferenceLimits, @monotonic_clock\.MonotonicClock\) -> LunaPreparedRequest raise RequestAdmissionError$' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'Luna request preparation must retain typed receipt/model/tokenizer binding' >&2
      failed=1
    fi
    if rg -n '^pub fn admit\(|AdmittedRequest' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' 'legacy request admission surface remains public' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn LunaPreparedRequest::take_claim\(Self\) -> LunaPreparedRequestClaim raise RequestAdmissionError$' \
      service/request_admission/pkg.generated.mbti ||
      rg -n 'claim_scheduler_request|LunaPreparedRequest::scheduler_request' \
        service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'prepared shell must destructively transfer its preallocated claim' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct (IncrementalRequestReceiver|ReceivedRequest|LunaPreparedRequest|LunaPreparedRequestClaim) \{\n  (?!// private fields)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' 'request admission owners must remain opaque' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct LunaPreparedRequest(?:Claim)? \{(?s:[^}]*)\} derive\([^)]*Debug' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'prepared shell and destructive claim must not expose Debug state' >&2
      failed=1
    fi
    if [ "$(rg -c '^pub fn LunaPreparedRequest::' \
      service/request_admission/pkg.generated.mbti)" != '8' ] ||
      ! rg -q '^pub fn LunaPreparedRequest::take_claim\(Self\) -> LunaPreparedRequestClaim raise RequestAdmissionError$' \
        service/request_admission/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaPreparedRequest::discard\(Self\) -> Unit raise RequestAdmissionError$' \
        service/request_admission/pkg.generated.mbti ||
      rg -n --pcre2 \
        '^pub fn LunaPreparedRequest::(receipt|receipt_at_millis|timestamp|deadline|admission_deadline|scheduler_request|is_stop_token|push_token|push_token_into|push_token_into_status|finish_into|finish_into_status|output_finished|output_stopped)\(' \
        service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'prepared shell must expose only binding preflight plus destructive transfer' >&2
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
        'prepared claim must expose only scheduler transfer, online output, and exact release' >&2
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
    if rg -n \
      '^pub fn (LunaPreparedRequest|LunaPreparedRequestClaim|LunaRequestPreparation[^:]*)::.*(LunaTokenBuffer|TokenBuffer|LunaIncrementalOutput(Workspace|Work|Lease)|Array\[Int\]|ArrayView\[Int\]|ReadOnlyArray\[Int\])' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'Luna preparation authority leaked raw token/output storage' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct (LunaRequestPreparationPool|LunaRequestPreparationAdmission|LunaRequestPreparationWork|LunaRequestPreparationStepBudget|LunaRequestPreparationWorkLimit|LunaRequestPreparationStorageBudget) \{\n  (?!// private fields)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' 'Luna preparation pool capabilities must remain opaque' >&2
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
      printf '%s\n' \
        'Luna preparation pool submit/progress/take evidence surface drifted' >&2
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
      printf '%s\n' \
        'Luna preparation pool progress vocabulary drifted' >&2
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
    if ! rg -q \
      '^pub fn IncrementalRequestReceiver::new\(@framed_wire\.FramedWireLimits, @monotonic_clock\.MonotonicClock\) -> Self' \
      service/request_admission/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn IncrementalRequestReceiver::append\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> Unit' \
        service/request_admission/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn IncrementalRequestReceiver::take_received\(Self\) -> ReceivedRequest' \
        service/request_admission/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn IncrementalRequestReceiver::next_capacity\(Self\) -> Int' \
        service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'trusted incremental request receiver surface drifted' >&2
      failed=1
    fi
    if [ "$(rg -c '^pub fn IncrementalRequestReceiver::' \
      service/request_admission/pkg.generated.mbti)" != '4' ]; then
      printf '%s\n' \
        'trusted incremental request receiver must expose exactly four methods' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn receive\(@framed_wire\.RequestFrameBuffer, FixedArray\[Byte\], Int, @monotonic_clock\.MonotonicClock\) -> ReceivedRequest' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'request admission must capture receipt before framed parsing' >&2
      failed=1
    fi
    if rg -q --pcre2 \
      '^pub (struct (RequestReceipt|PendingRequestReceipt)|fn (capture_receipt|begin_receipt|bind_received|ReceivedRequest::(receipt|deadline|frame|request)|IncrementalRequestReceiver::(receipt|receipt_at_millis|timestamp|deadline|frame|request)))' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'request receipt evidence must not escape framed receipt admission' >&2
      failed=1
    fi
  fi
  if ! rg -q --pcre2 -U \
    'pub fn receive(?s).*let receipt = capture_receipt_with_clock.*let frame = buffer\.load.*bind_received\(receipt, frame, buffer\.inference_limits\(\)\)' \
    service/request_admission/receipt.mbt; then
    printf '%s\n' \
      'legacy request receipt must capture before authoritative load/bind' >&2
    failed=1
  fi
  if ! rg -q --pcre2 -U \
    'fn IncrementalRequestReceiver::preflight_append(?s).*source_offset < 0.*let capacity = self\.reader\.next_capacity.*if length > capacity' \
    service/request_admission/receipt.mbt ||
    ! rg -q --pcre2 -U \
      'pub fn IncrementalRequestReceiver::append(?s).*self\.preflight_append.*self\.clock\.now_millis.*self\.receipt_at_millis = at_millis.*self\.has_receipt = true.*self\.append_validated' \
      service/request_admission/receipt.mbt; then
    printf '%s\n' \
      'incremental request receipt must validate before clock and store before copy' >&2
    failed=1
  fi
  incremental_reader_calls="$(rg -l \
    'IncrementalRequestReader::new\(' \
    --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
    . | sed 's#^\./##' | sort || true)"
  if [ "$incremental_reader_calls" != \
    $'service/framed_wire/request_reader.mbt\nservice/request_admission/receipt.mbt' ]; then
    printf '%s\n' \
      'incremental framed reader must be composed only by trusted receipt admission' >&2
    failed=1
  fi
  incremental_reader_types="$(rg -l \
    'IncrementalRequestReader' \
    --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
    . | sed 's#^\./##' | sort || true)"
  if [ "$incremental_reader_types" != \
    $'service/framed_wire/request_reader.mbt\nservice/framed_wire/types.mbt\nservice/request_admission/receipt.mbt\nservice/request_admission/types.mbt' ]; then
    printf '%s\n' \
      'incremental framed reader type must not escape its trusted composition' >&2
    failed=1
  fi
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
