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

# Semantic Luna event storage is a bounded typed leaf. Public views are opaque
# epoch capabilities: tuple constructors or owner/epoch accessors would let an
# external caller forge retirement authority.
if [ -d service/luna_event ]; then
  fail_matches \
    'Luna semantic event imports outside inference contracts + model spec:' \
    --pcre2 --glob 'service/luna_event/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|model/spec")'
  fail_matches \
    'Luna semantic event storage must remain synchronous and native-ABI free:' \
    --glob 'service/luna_event/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/luna_event/pkg.generated.mbti ]; then
    for luna_view in \
      LunaEventView \
      LunaAcceptedEventView \
      LunaTokenEventView \
      LunaUsageEventView \
      LunaCompletedEventView \
      LunaFailedEventView; do
      if ! rg -U -q \
        "^pub struct ${luna_view} \\{\\n  // private fields\\n\\}" \
        service/luna_event/pkg.generated.mbti; then
        printf '%s\n' \
          "Luna semantic view is not opaque in generated interface: ${luna_view}" >&2
        failed=1
      fi
    done
    if ! rg -U -q \
      '^pub struct LunaEventOwner \{\n  // private fields\n\}' \
      service/luna_event/pkg.generated.mbti; then
      printf '%s\n' \
        'Luna semantic event owner fields escaped its generated interface' >&2
      failed=1
    fi
    if rg -n \
      '^pub struct Luna(Event|AcceptedEvent|TokenEvent|UsageEvent|CompletedEvent|FailedEvent)View\(|^pub fn Luna(Event|AcceptedEvent|TokenEvent|UsageEvent|CompletedEvent|FailedEvent)View::(owner|epoch|raw)' \
      service/luna_event/pkg.generated.mbti; then
      printf '%s\n' \
        'Luna semantic view exposes a forgeable tuple or raw owner/epoch accessor' >&2
      failed=1
    fi
  fi
fi

# Completion wire tags are private codec details. Consumers receive the
# canonical worker-protocol enums and cannot branch on duplicated numeric tags.
if [ -f engine/worker_wire/pkg.generated.mbti ]; then
  if rg -n '^pub fn CompletionFrameEntry::(kind_value|failure_value)\(' \
    engine/worker_wire/pkg.generated.mbti; then
    printf '%s\n' \
      'completion-frame entry must not expose raw numeric tag accessors' >&2
    failed=1
  fi
  if ! rg -q \
      '^pub fn CompletionFrameEntry::kind\(Self\) -> @worker_protocol\.CompletionEntryKind raise WorkerWireError$' \
      engine/worker_wire/pkg.generated.mbti ||
    ! rg -q \
      '^pub fn CompletionFrameEntry::failure\(Self\) -> @worker_protocol\.WorkerFailure raise WorkerWireError$' \
      engine/worker_wire/pkg.generated.mbti; then
    printf '%s\n' \
      'completion-frame entry must expose typed kind and failure accessors' >&2
    failed=1
  fi
  completion_raw_tag_calls=$(rg -n '\.(kind_value|failure_value)\(' \
    engine/worker_wire engine/worker_process scheduler/core \
    tests/device_worker_alloc --glob '*.mbt' 2>/dev/null || true)
  if [ -n "$completion_raw_tag_calls" ]; then
    printf '%s\n%s\n' \
      'completion-frame consumers must not use raw numeric tag accessors:' \
      "$completion_raw_tag_calls" >&2
    failed=1
  fi
fi

# Canonical service frames are bounded contract codecs only. Transport,
# scheduling, tokenization, filesystem, and engine composition belong above
# this leaf package.
if [ -d service/framed_wire ]; then
  fail_matches \
    'framed service wire imports outside contracts/inference + model/spec + Luna events:' \
    --pcre2 --glob 'service/framed_wire/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|model/spec"|service/luna_event")'
  fail_matches \
    'framed service wire must remain synchronous and native-ABI free:' \
    --glob 'service/framed_wire/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/framed_wire/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn RequestFrameBuffer::load\(Self, FixedArray\[Byte\], Int\)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'framed request decoder must retain its fixed-buffer API' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn EventFrameBuffer::load\(Self, FixedArray\[Byte\], Int\)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'framed event decoder must retain its fixed-buffer API' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn IncrementalRequestReader::new\(FramedWireLimits\) -> Self' \
      service/framed_wire/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn IncrementalRequestReader::append\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> Unit' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn IncrementalRequestReader::next_capacity\(Self\) -> Int' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn IncrementalRequestReader::take\(Self\) -> ValidatedRequestFrame' \
        service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental framed request reader surface drifted' >&2
      failed=1
    fi
    if [ "$(rg -c '^pub fn IncrementalRequestReader::' \
      service/framed_wire/pkg.generated.mbti)" != '4' ]; then
      printf '%s\n' \
        'incremental framed reader must expose exactly four methods' >&2
      failed=1
    fi
    if rg -q \
      '^pub fn IncrementalRequestReader::new\(@?framed_wire\.RequestFrameBuffer\)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental framed reader must not accept a caller-retained buffer' >&2
      failed=1
    fi
    if rg -q --pcre2 \
      '^pub fn IncrementalRequestReader::(storage|scratch|declared_length|frame|request)\(' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental framed reader must not expose raw storage or frame evidence' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct (FramedWireLimits|RequestFrameBuffer|IncrementalRequestReader|EventFrameBuffer|ValidatedRequestFrame|ValidatedEventFrame) \{\n  (?!// private fields)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'framed wire owner and limits fields must remain private' >&2
      failed=1
    fi
    if ! rg -q '^pub fn LunaFramedEventAdapter::new\(FramedWireLimits\) -> Self raise FramedWireError$' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaFramedEventAdapter::stage\(Self, @luna_event\.LunaEventView\) -> Unit raise FramedWireError$' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaFramedEventAdapter::length\(Self\) -> Int raise FramedWireError$' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaFramedEventAdapter::copy_to\(Self, FixedArray\[Byte\], destination_offset~ : Int\) -> Int raise FramedWireError$' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaFramedEventAdapter::release\(Self\) -> Unit raise FramedWireError$' \
        service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'Luna framed event adapter surface drifted' >&2
      failed=1
    fi
    if ! rg -q --pcre2 -U \
        'pub struct LunaFramedEventAdapter \{\n  // private fields\n\}' \
        service/framed_wire/pkg.generated.mbti ||
      rg -n '^pub fn LunaFramedEventAdapter::(ack|retire|view|event)\(' \
        service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' \
        'framed Luna adapter must remain opaque and must not receive ACK authority' >&2
      failed=1
    fi
  fi
  if ! rg -q --pcre2 -U \
    'pub fn IncrementalRequestReader::append(?s).*let declared = self\.preflight_prefix.*for index in 0\.\.<length.*self\.frame_buffer\.load' \
    service/framed_wire/request_reader.mbt ||
    ! rg -q '16 - self\.length' service/framed_wire/request_reader.mbt; then
    printf '%s\n' \
      'incremental framed reader must authenticate the exact prefix before copy/load' >&2
    failed=1
  fi
fi

# The bounded Luna tokenizer worker is synchronous cooperative work. It owns
# only fixed scratch and resumable scalar state; async orchestration, sockets,
# process authority, and dynamically growing work storage stay above it.
if [ -d tokenizer ]; then
  fail_matches \
    'Luna tokenizer work acquired async, transport, process, or native authority:' \
    --glob 'tokenizer/luna_worker*.mbt' --glob 'tokenizer/moon.pkg' \
    'moonbitlang/async|socket|internal/process|worker_process|extern\s+"[cC]"|#external|pub async fn'
  fail_matches \
    'Luna tokenizer warmed work uses dynamically growing Array or Map storage:' \
    --pcre2 --glob 'tokenizer/luna_worker*.mbt' \
    '(^|[^A-Za-z])((Array|Map)\[|Array::|Map\()'
  if ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerWork \{\n  priv worker : LunaTokenizerWorker\n  priv epoch : UInt64\n\}' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerStepBudget \{\n  priv work_units : Int\n\}' \
      tokenizer/luna_worker_types.mbt ||
    rg -n --pcre2 -U \
      'pub struct LunaTokenizer(?:Work|Worker|StepBudget)(?s:.*?)derive\([^)]*Debug' \
      tokenizer/luna_worker*.mbt; then
    printf '%s\n' \
      'Luna tokenizer work, worker, and budget must remain opaque without Debug' >&2
    failed=1
  fi
  worker_api="$(rg --no-filename -o '^pub fn LunaTokenizerWorker::[a-z_]+' \
    tokenizer/luna_worker*.mbt | sort)"
  if [ "$worker_api" != \
    $'pub fn LunaTokenizerWorker::begin_bytes\npub fn LunaTokenizerWorker::new' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer worker must expose only construction and epoch-bound begin:' \
      "$worker_api" >&2
    failed=1
  fi
  work_api="$(rg --no-filename -o '^pub fn LunaTokenizerWork::[a-z_]+' \
    tokenizer/luna_worker*.mbt | sort)"
  if [ "$work_api" != \
    $'pub fn LunaTokenizerWork::abort\npub fn LunaTokenizerWork::copy_tokens_to\npub fn LunaTokenizerWork::last_work_units\npub fn LunaTokenizerWork::progress\npub fn LunaTokenizerWork::token_count\npub fn LunaTokenizerWork::was_truncated' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer work capability surface drifted:' "$work_api" >&2
    failed=1
  fi
  if [ -f tokenizer/pkg.generated.mbti ]; then
    if [ "$(rg -c '^pub fn LunaTokenizerWorker::' \
        tokenizer/pkg.generated.mbti)" != '2' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerWork::' \
        tokenizer/pkg.generated.mbti)" != '6' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerStepBudget::' \
        tokenizer/pkg.generated.mbti)" != '2' ] ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerStepBudget \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerWork \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerWorker \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      rg -q --pcre2 \
        '^pub struct LunaTokenizer(?:StepBudget|Work|Worker)\(|^pub fn LunaTokenizer(?:StepBudget|Work|Worker)::[^\n]*(?:owner|epoch|storage)|LunaTokenizer(?:StepBudget|Work|Worker).*derive\([^)]*Debug' \
        tokenizer/pkg.generated.mbti; then
      printf '%s\n' \
        'generated Luna tokenizer authorities must remain exact and opaque' >&2
      failed=1
    fi
    if ! rg -q --pcre2 -U \
        'pub\(all\) enum LunaTokenizerProgress \{\n  LunaTokenizerPending\n  LunaTokenizerReady\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::begin_bytes\(Self, Bytes, special_policy~ : SpecialTokenPolicy, output_limit~ : Int, truncation~ : TruncationPolicy\) -> LunaTokenizerWork raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::new\(TokenizerSpec, input_capacity~ : Int, output_capacity~ : Int, LunaTokenizerStepBudget\) -> Self raise TokenizerError$' \
        tokenizer/pkg.generated.mbti; then
      printf '%s\n' \
        'generated Luna tokenizer lifecycle surface drifted' >&2
      failed=1
    fi
  fi
  if ! rg -q 'const LUNA_TOKENIZER_MAX_STEP_WORK_UNITS : Int = 65536' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q 'length > worker\.budget\.work_units()' tokenizer/luna_worker.mbt ||
    ! rg -q --pcre2 -U \
      'pub fn LunaTokenizerWork::progress(?s).*let maximum = worker\.budget\.work_units\(\).*while used < maximum' \
      tokenizer/luna_worker_progress.mbt;
  then
    printf '%s\n' \
      'Luna tokenizer progress/copy work bounds are not enforced at the owner' >&2
    failed=1
  fi
fi

# Incremental output owns fixed per-request decode/matcher state only. Socket,
# scheduler, worker, filesystem, and native authority stay outside this leaf.
if [ -d service/incremental_output ]; then
  fail_matches \
    'incremental output imports outside inference contracts + tokenizer:' \
    --pcre2 --glob 'service/incremental_output/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|tokenizer")'
  fail_matches \
    'incremental output must remain synchronous and native-ABI free:' \
    --glob 'service/incremental_output/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/incremental_output/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn IncrementalOutput::push_token_into\(Self, Int, FixedArray\[Byte\], destination_offset~ : Int\)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output must retain its fixed-destination token API' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct IncrementalOutput \{\n  (?!// private fields)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output state must remain opaque' >&2
      failed=1
    fi
  fi
fi

# Request admission is the synchronous tokenizer-worker bridge. It may bind
# contracts, tokenizer, monotonic time, incremental output, and the scheduler
# request value, but it must not acquire transport/process/device authority or
# become an async listener.
if [ -d service/request_admission ]; then
  fail_matches \
    'request admission imports a forbidden engine or transport owner:' \
    --glob 'service/request_admission/moon.pkg' \
    'worker_service|worker_process|internal/process|moonbitlang/async|runtime/approved_fs'
  fail_matches \
    'request admission must remain synchronous and native-ABI free:' \
    --glob 'service/request_admission/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
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
      service/request_admission/pkg.generated.mbti)" != '7' ] ||
      ! rg -q '^pub fn LunaPreparedRequest::take_claim\(Self\) -> LunaPreparedRequestClaim raise RequestAdmissionError$' \
        service/request_admission/pkg.generated.mbti ||
      rg -n --pcre2 \
        '^pub fn LunaPreparedRequest::(receipt|receipt_at_millis|timestamp|deadline|admission_deadline|scheduler_request|is_stop_token|push_token|push_token_into|push_token_into_status|finish_into|finish_into_status|output_finished|output_stopped)\(' \
        service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'prepared shell must expose only binding preflight plus destructive transfer' >&2
      failed=1
    fi
    if [ "$(rg -c '^pub fn LunaPreparedRequestClaim::' \
      service/request_admission/pkg.generated.mbti)" != '4' ] ||
      ! rg -q '^pub fn LunaPreparedRequestClaim::scheduler_request\(Self\) -> @core\.TokenizedRequest$' \
        service/request_admission/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaPreparedRequestClaim::is_stop_token\(Self, Int\) -> Bool$' \
        service/request_admission/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaPreparedRequestClaim::push_token_into_status\(' \
        service/request_admission/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaPreparedRequestClaim::finish_into_status\(' \
        service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'prepared claim must expose only scheduler transfer and online output operations' >&2
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
    'pub fn receive(?s).*let receipt = capture_receipt_with_clock.*let frame = buffer\.load.*bind_received\(receipt, frame\)' \
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

printf '%s\n' 'LunaFlux service and wire boundaries are valid.'
