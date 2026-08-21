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
    if ! rg -q '^pub fn LunaFramedRequestStepBudget::new\(Int\) -> Self raise FramedWireError$' \
        service/framed_wire/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaFramedRequestStepBudget::work_units\(Self\) -> Int$' \
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
      printf '%s\n' 'Luna framed-request cooperative surface drifted' >&2
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
      printf '%s\n' 'Luna framed-request capability method set drifted' >&2
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
    if [ "$actual_luna_view_methods" != "$expected_luna_view_methods" ]; then
      printf '%s\n' 'Luna framed-request scalar view surface drifted' >&2
      failed=1
    fi
    luna_request_private_count="$(rg -c --pcre2 -U \
      'pub struct LunaFramedRequest(StepBudget|Workspace|Work|View) \{\n  // private fields\n\}' \
      service/framed_wire/pkg.generated.mbti)"
    if [ "$luna_request_private_count" != '4' ] ||
      rg -n '^pub fn LunaFramedRequest(StepBudget|Workspace|Work|View)::.*-> .*(FixedArray|Array\[|ArrayView|ReadOnlyArray|Bytes|String|GenerateRequest|ValidatedRequestFrame)' \
        service/framed_wire/pkg.generated.mbti ||
      rg -n '^pub fn LunaFramedRequest(StepBudget|Workspace|Work|View)::(owner|epoch|storage|workspace)\(' \
        service/framed_wire/pkg.generated.mbti ||
      rg -n 'pub struct LunaFramedRequest(StepBudget|Workspace|Work|View).*derive\([^)]*Debug' \
        service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' \
        'Luna framed-request capabilities leaked storage or representation' >&2
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
      printf '%s\n' 'Luna framed-event cooperative surface drifted' >&2
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
    --glob 'tokenizer/luna_worker*.mbt' \
    --glob 'tokenizer/luna_input_write.mbt' --glob 'tokenizer/moon.pkg' \
    'moonbitlang/async|socket|internal/process|worker_process|extern\s+"[cC]"|#external|pub async fn'
  fail_matches \
    'Luna tokenizer warmed work uses dynamically growing Array or Map storage:' \
    --pcre2 --glob 'tokenizer/luna_worker*.mbt' \
    --glob 'tokenizer/luna_input_write.mbt' \
    '(^|[^A-Za-z])((Array|Map)\[|Array::|Map\()'
  if ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerInputWrite \{\n  priv worker : LunaTokenizerWorker\n  priv epoch : UInt64\n\}' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerWork \{\n  priv worker : LunaTokenizerWorker\n  priv epoch : UInt64\n\}' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerStepBudget \{\n  priv work_units : Int\n\}' \
      tokenizer/luna_worker_types.mbt ||
    rg -n --pcre2 -U \
      'pub struct LunaTokenizer(?:InputWrite|Work|Worker|StepBudget)(?s:.*?)derive\([^)]*Debug' \
      tokenizer/luna_worker*.mbt tokenizer/luna_input_write.mbt; then
    printf '%s\n' \
      'Luna tokenizer work, worker, and budget must remain opaque without Debug' >&2
    failed=1
  fi
  worker_api="$(rg --no-filename -o '^pub fn LunaTokenizerWorker::[a-z_]+' \
    tokenizer/luna_worker*.mbt tokenizer/luna_input_write.mbt | sort)"
  if [ "$worker_api" != \
    $'pub fn LunaTokenizerWorker::begin_bytes\npub fn LunaTokenizerWorker::begin_luna_input\npub fn LunaTokenizerWorker::new\npub fn LunaTokenizerWorker::required_byte_cells\npub fn LunaTokenizerWorker::required_int_cells' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer worker must expose only construction and epoch-bound begin:' \
      "$worker_api" >&2
    failed=1
  fi
  input_write_api="$(rg --no-filename -o '^pub fn LunaTokenizerInputWrite::[a-z_]+' \
    tokenizer/luna_input_write.mbt | sort)"
  if [ "$input_write_api" != \
    $'pub fn LunaTokenizerInputWrite::abort\npub fn LunaTokenizerInputWrite::finish\npub fn LunaTokenizerInputWrite::push_byte' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer input-write capability surface drifted:' \
      "$input_write_api" >&2
    failed=1
  fi
  work_api="$(rg --no-filename -o '^pub fn LunaTokenizerWork::[a-z_]+' \
    tokenizer/luna_worker*.mbt | sort)"
  if [ "$work_api" != \
    $'pub fn LunaTokenizerWork::abort\npub fn LunaTokenizerWork::copy_tokens_to\npub fn LunaTokenizerWork::last_work_units\npub fn LunaTokenizerWork::progress\npub fn LunaTokenizerWork::token_count\npub fn LunaTokenizerWork::token_status\npub fn LunaTokenizerWork::was_truncated' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer work capability surface drifted:' "$work_api" >&2
    failed=1
  fi
  if [ -f tokenizer/pkg.generated.mbti ]; then
    if [ "$(rg -c '^pub fn LunaTokenizerWorker::' \
        tokenizer/pkg.generated.mbti)" != '5' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerInputWrite::' \
        tokenizer/pkg.generated.mbti)" != '3' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerWork::' \
        tokenizer/pkg.generated.mbti)" != '7' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerStepBudget::' \
        tokenizer/pkg.generated.mbti)" != '2' ] ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerStepBudget \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerInputWrite \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerWork \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerWorker \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      rg -q --pcre2 \
        '^pub struct LunaTokenizer(?:InputWrite|StepBudget|Work|Worker)\(|^pub fn LunaTokenizer(?:InputWrite|StepBudget|Work|Worker)::[^\n]*(?:owner|epoch|storage)|LunaTokenizer(?:InputWrite|StepBudget|Work|Worker).*derive\([^)]*Debug' \
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
      ! rg -q '^pub fn LunaTokenizerWorker::begin_luna_input\(Self, byte_count~ : Int, special_policy~ : SpecialTokenPolicy, output_limit~ : Int, truncation~ : TruncationPolicy\) -> LunaTokenizerInputWrite raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::required_byte_cells\(Int\) -> UInt64 raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::new\(TokenizerSpec, input_capacity~ : Int, output_capacity~ : Int, LunaTokenizerStepBudget\) -> Self raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerInputWrite::abort\(Self\) -> Unit raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerInputWrite::finish\(Self\) -> LunaTokenizerWork raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerInputWrite::push_byte\(Self, Byte\) -> Unit raise TokenizerError$' \
        tokenizer/pkg.generated.mbti; then
      printf '%s\n' \
        'generated Luna tokenizer lifecycle surface drifted' >&2
      failed=1
    fi
  fi
  if ! rg -q 'priv input : FixedArray\[Byte\]' \
      tokenizer/luna_worker_types.mbt ||
    rg -n 'priv (mut )?input : Bytes' tokenizer/luna_worker_types.mbt ||
    rg -n '^pub fn LunaTokenizer(?:Worker|InputWrite)::(?:input|source|raw_bytes|request_bytes)' \
      tokenizer/pkg.generated.mbti ||
    ! rg -q 'self\.input_offset .* self\.input_length' \
      tokenizer/luna_worker_progress.mbt; then
    printf '%s\n' \
      'Luna tokenizer worker no longer proves fixed owned byte input' >&2
    failed=1
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
    if ! rg -q \
      '^pub fn LunaIncrementalOutputWorkspace::begin_luna_request_semantics\(Self, @inference\.LunaRequestSemanticView, special_policy~ : @tokenizer\.DecodeSpecialPolicy\) -> LunaIncrementalOutputWork raise IncrementalOutputError$' \
      service/incremental_output/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn LunaIncrementalOutputWorkspace::required_reference_cells\(@inference\.InferenceLimits\) -> Int raise IncrementalOutputError$' \
        service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output semantic-view setup surface drifted' >&2
      failed=1
    fi
    if rg -n \
      '^pub fn LunaIncrementalOutput[^ ]*::.*@inference\.LunaRequestSemantic(Storage|Write|Work|Lease)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output acquired semantic storage or release authority' >&2
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
    if [ "$claim_scheduler_consumers" != \
      'service/online_session/admission.mbt' ]; then
      printf '%s\n' \
        'prepared scheduler-request borrow escaped the online admission bridge' >&2
      failed=1
    fi
    direct_claim_releases="$(rg -n 'claim\.release\(\)' \
      service/online_session --glob '*.mbt' | wc -l | tr -d ' ')"
    lifecycle_claim_releases="$(rg -n 'self\.release_request_claim\(\)' \
      service/online_session/lifecycle.mbt | wc -l | tr -d ' ')"
    if [ "$direct_claim_releases" != '2' ] ||
      [ "$lifecycle_claim_releases" != '2' ] ||
      ! rg -q --pcre2 -U \
        'self\.lease_owner\(\)\.admit\(claim\.scheduler_request\(\)\) catch \{[\s\S]*try! claim\.release\(\)[\s\S]*try! self\.events\.discard\(\)' \
        service/online_session/admission.mbt ||
      ! rg -q --pcre2 -U \
        'fn LunaOnlineInstance::close_terminal_owner[\s\S]*lease\.close_terminal\(\)[\s\S]*lease\.retry_close_terminal\(\)[\s\S]*self\.release_request_claim\(\)[\s\S]*self\.reset_request\(\)' \
        service/online_session/lifecycle.mbt ||
      ! rg -q --pcre2 -U \
        'lease\.retire_terminal_request\(\) catch[\s\S]*self\.release_request_claim\(\)[\s\S]*self\.reset_request\(\)' \
        service/online_session/lifecycle.mbt; then
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
      ! rg -q '^pub fn LunaRequestPreparationWork::take_prepared\(Self\) -> LunaPreparedRequest raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaRequestPreparationWork::last_work_units\(Self\) -> Int raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaRequestPreparationWork::total_work_units\(Self\) -> UInt64 raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'Luna preparation pool submit/progress/take evidence surface drifted' >&2
      failed=1
    fi
    if rg -n \
        '^pub fn .*(@framed_wire\.LunaFramedRequest(Workspace|Work|View)|RequestReceipt|@inference\.AdmissionDeadline)' \
        service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'direct framed preparation leaked scanner, receipt, or deadline authority' >&2
      failed=1
    fi
    if rg -n \
        '^pub (struct|enum) (Luna.*Framed.*(Receipt|Admission)|RequestReceipt)' \
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

assert_self_factory_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$(rg \
    "^pub fn ${authority_type}::.* -> Self( raise .*)?$" \
    "$interface" | sed \
    "s/^pub fn ${authority_type}:://; s/(.*$//" | sort || true)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public Self factory allowlist drifted" >&2
    failed=1
  fi
}

assert_authority_escape_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$({
    rg "^pub fn ${authority_type}::.* -> Self( raise .*)?$" \
      "$interface" || true
    rg --pcre2 \
      "^pub fn .* -> .*(?<![A-Za-z0-9_])${authority_type}(?![A-Za-z0-9_]).*$" \
      "$interface" || true
  } | sed 's/^pub fn //; s/(.*$//' | sort -u)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public authority escape allowlist drifted" >&2
    failed=1
  fi
}

assert_authority_method_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$(rg "^pub fn ${authority_type}::" "$interface" | sed \
    "s/^pub fn ${authority_type}:://; s/(.*$//" | sort || true)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public method allowlist drifted" >&2
    failed=1
  fi
}

for request_authority in \
    AdmissionDeadline \
    CachePolicy \
    CacheScope \
    DeadlineBudget \
    EffectiveLimits \
    GenerateRequest \
    InferenceLimits \
    ProtocolVersion \
    RequestId \
    SamplingParameters \
    SamplingSeed \
    StopConditions \
    TextInput \
    TokenBuffer \
    TraceCorrelation; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${request_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference request authority ${request_authority} is not opaque" >&2
    failed=1
  fi
done
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti AdmissionDeadline from_receipt
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti CachePolicy new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti CacheScope from_ascii
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti DeadlineBudget from_milliseconds
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti EffectiveLimits new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti GenerateRequest new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti InferenceLimits new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti Input 'from_token_ids
from_utf8'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti ProtocolVersion 'from_wire
v1'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti RequestId new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti SamplingParameters 'greedy
sample'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti SamplingSeed new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti StopConditions 'new
token_only'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TextInput from_utf8
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TokenBuffer new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TraceCorrelation from_ascii
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti AdmissionDeadline AdmissionDeadline::from_receipt
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti CachePolicy 'CachePolicy::new
GenerateRequest::cache'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti CacheScope 'CachePolicy::scope
CacheScope::from_ascii'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti DeadlineBudget 'DeadlineBudget::from_milliseconds
GenerateRequest::deadline'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti EffectiveLimits 'AcceptedEvent::effective_limits
EffectiveLimits::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti GenerateRequest GenerateRequest::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti InferenceLimits 'InferenceLimits::new
LunaRequestSemanticView::inference_limits'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti Input 'GenerateRequest::input
Input::from_token_ids
Input::from_utf8'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti ProtocolVersion 'GenerateRequest::protocol_version
ProtocolVersion::from_wire
ProtocolVersion::v1'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti RequestId 'GenerateRequest::request_id
RequestId::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti SamplingParameters 'GenerateRequest::sampling
SamplingParameters::greedy
SamplingParameters::sample'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti SamplingSeed 'GenerateRequest::sampling_seed
SamplingSeed::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti StopConditions 'GenerateRequest::stops
StopConditions::new
StopConditions::token_only'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TextInput TextInput::from_utf8
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TokenBuffer 'LunaTokenBufferLease::token_buffer
TokenBuffer::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TraceCorrelation 'GenerateRequest::trace
TraceCorrelation::from_ascii'
if ! rg -q --pcre2 -U \
    '^pub\(all\) enum Input \{\n  TokenIds\(TokenBuffer\)\n  Text\(TextInput\)\n\}$' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference Input authority variants drifted' >&2
  failed=1
fi
if ! rg -F -x -q \
    'pub fn Input::from_token_ids(ArrayView[Int], InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn Input::from_utf8(BytesView, InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn GenerateRequest::input(Self) -> Input' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn TokenBuffer::new(ArrayView[Int], InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn LunaTokenBufferLease::token_buffer(Self) -> TokenBuffer raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference input and token authority signatures drifted' >&2
  failed=1
fi
if rg -q --pcre2 -U \
    '^pub struct (CachePolicy|CacheScope|GenerateRequest|SamplingSeed|StopConditions|TextInput|TokenBuffer|TraceCorrelation) \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'payload-bearing inference request authority became debuggable' >&2
  failed=1
fi

for semantic_authority in \
    LunaRequestSemanticFailure \
    LunaRequestSemanticLease \
    LunaRequestSemanticProgress \
    LunaRequestSemanticState \
    LunaRequestSemanticStepBudget \
    LunaRequestSemanticStopTokenStatus \
    LunaRequestSemanticStorage \
    LunaRequestSemanticView \
    LunaRequestSemanticWork \
    LunaRequestSemanticWrite; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${semantic_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference semantic authority ${semantic_authority} is not opaque" >&2
    failed=1
  fi
done
if rg -q --pcre2 -U \
    '^pub struct LunaRequestSemantic[^ ]* \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority became debuggable' >&2
  failed=1
fi
if rg -n --pcre2 \
    '^pub fn LunaRequestSemantic[^:]*::.* -> .*\b(Array|ArrayView|ReadOnlyArray|FixedArray|Bytes|BytesView|String|StringView)\b' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority exposed a raw payload collection' >&2
  failed=1
fi
if rg -n '^pub fn LunaRequestSemantic[^:]*::.*epoch' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference semantic authority exposed a raw epoch' >&2
  failed=1
fi
if rg 'LunaRequestSemantic(Failure|Lease|Progress|State|StepBudget|StopTokenStatus|Storage|View|Work|Write)' \
    contracts/inference/pkg.generated.mbti | \
    rg -n -v '^pub (struct|fn) '; then
  printf '%s\n' \
    'inference semantic authority was embedded in another public type' >&2
  failed=1
fi
if rg -n --pcre2 \
    '^pub fn (?!LunaRequestSemantic[^:]*::)[^(]*\([^)]*LunaRequestSemantic(Failure|Lease|Progress|State|StepBudget|StopTokenStatus|Storage|View|Work|Write)[^)]*\)' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority gained a non-owner public consumer' >&2
  failed=1
fi

assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticFailure 'field
index
issue'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticLease 'release
stop_token_view
view'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticProgress 'is_failed
is_pending
is_ready'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticState 'is_failed
is_ready
is_validating'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStepBudget 'new
work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStopTokenStatus 'is_absent
is_present
is_stale
work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStorage 'begin
new
required_byte_cells
required_int_cells'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticView 'cache_permission
cache_scope_byte_at
cache_scope_length
inference_limits
is_stop_token
stop_string_byte_at
stop_string_count
stop_string_length
stop_token_at
stop_token_count'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWork 'abort
failure
last_work_units
progress
state
take_lease
total_work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWrite 'abort
finish
push_cache_scope_byte
push_stop_string_byte
push_stop_string_length
push_stop_token'

assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticFailure \
  LunaRequestSemanticWork::failure
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticLease \
  LunaRequestSemanticWork::take_lease
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticProgress \
  LunaRequestSemanticWork::progress
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticState \
  LunaRequestSemanticWork::state
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStepBudget \
  LunaRequestSemanticStepBudget::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStopTokenStatus \
  $'LunaRequestSemanticView::is_stop_token\nLunaRequestStopTokenRetentionSlot::is_stop_token\nLunaRequestStopTokenView::is_stop_token'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStorage \
  LunaRequestSemanticStorage::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticView \
  LunaRequestSemanticLease::view
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWork \
  LunaRequestSemanticWrite::finish
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWrite \
  LunaRequestSemanticStorage::begin

for stop_authority in \
    LunaRequestStopTokenRetentionSlot \
    LunaRequestStopTokenView; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${stop_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference token-stop authority ${stop_authority} is not opaque" >&2
    failed=1
  fi
done
if rg -q --pcre2 -U \
    '^pub struct LunaRequestStopToken[^ ]* \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference token-stop authority became debuggable' >&2
  failed=1
fi
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenRetentionSlot 'is_live
is_stop_token
new
release'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenView 'is_live
is_stop_token
retain_into'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenRetentionSlot \
  LunaRequestStopTokenRetentionSlot::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenView \
  LunaRequestSemanticLease::stop_token_view

for model_identity_authority in \
  ContentDigest \
  LlamaModelMetadata \
  LlamaModelSpec \
  ModelIdentity \
  PlanDigest; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${model_identity_authority} \\{\\n  // private fields\\n\\}" \
      model/spec/pkg.generated.mbti; then
    printf '%s\n' \
      "model identity authority ${model_identity_authority} is not opaque" >&2
    failed=1
  fi
done
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti ContentDigest from_sha256
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti LlamaModelMetadata from_verified_content
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti LlamaModelSpec new
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti ModelIdentity new
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti PlanDigest from_sha256
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti ContentDigest 'ContentDigest::from_sha256
ModelIdentity::content'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti LlamaModelMetadata LlamaModelMetadata::from_verified_content
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti LlamaModelSpec 'LlamaModelMetadata::spec
LlamaModelSpec::new'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti ModelIdentity 'LlamaModelMetadata::identity
ModelIdentity::new'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti PlanDigest 'LlamaModelSpec::canonical_paged_plan_digest
LlamaModelSpec::canonical_plan_digest
ModelIdentity::plan
PlanDigest::from_sha256'
if rg -n '^pub(\(all\))? type' \
    contracts/inference/pkg.generated.mbti model/spec/pkg.generated.mbti; then
  printf '%s\n' \
    'authority packages must not hide authority returns behind public aliases' >&2
  failed=1
fi
if ! rg -q '^pub fn ContentDigest::from_sha256\(String\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn PlanDigest::from_sha256\(String\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn ModelIdentity::new\(ContentDigest, PlanDigest\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelMetadata::from_verified_content\(LlamaModelSpec, ContentDigest\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelSpec::new\(vocabulary_size~ : Int, hidden_size~ : Int, intermediate_size~ : Int, layer_count~ : Int, attention_head_count~ : Int, kv_head_count~ : Int, context_length~ : Int, rope_theta~ : Double, rms_norm_epsilon\? : Double, weight_dtype~ : ModelDType, kv_dtype~ : ModelDType\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti; then
  printf '%s\n' 'model authority factory exclusivity drifted' >&2
  failed=1
fi

if [ -f contracts/inference/pkg.generated.mbti ] &&
  [ -f scheduler/core/pkg.generated.mbti ]; then
  if rg -n '^pub fn TokenBuffer::token_ids\(|^pub fn TokenizedRequest::(input_tokens|input_token_at)\(' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
    rg -n --pcre2 -U \
      'pub struct (TokenBuffer|TokenizedRequest) \{\n  (?!// private fields)' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
    rg -n --pcre2 -U \
      'pub struct (TokenBuffer|TokenizedRequest) \{(?s:[^}]*)\} derive\([^)]*Debug' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
    printf '%s\n' \
      'tokenized scheduler requests must remain opaque without raw token arrays' >&2
    failed=1
  fi

  token_bound_surface="$(rg \
    '^pub fn (LunaTokenBufferBoundStatus::|TokenBuffer::maximum_token_status\()' \
    contracts/inference/pkg.generated.mbti || true)"
  if ! rg -q --pcre2 -U \
      '^pub struct LunaTokenBufferBoundStatus \{\n  // private fields\n\}$' \
      contracts/inference/pkg.generated.mbti ||
    rg -q --pcre2 -U \
      '^pub struct LunaTokenBufferBoundStatus \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
      contracts/inference/pkg.generated.mbti ||
    [ "$token_bound_surface" != \
    $'pub fn LunaTokenBufferBoundStatus::is_out_of_range(Self) -> Bool\npub fn LunaTokenBufferBoundStatus::is_stale(Self) -> Bool\npub fn LunaTokenBufferBoundStatus::is_within_bound(Self) -> Bool\npub fn TokenBuffer::maximum_token_status(Self, Int) -> LunaTokenBufferBoundStatus' ]; then
    printf '%s\n' \
      'authenticated Luna token-maximum bound opacity or surface drifted' >&2
    failed=1
  fi

  if rg -n \
      '^pub fn TokenBuffer::.*maximum.* -> Int$|^pub fn TokenizedRequest::input_maximum_token_status\(' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
    printf '%s\n' \
      'raw token maximum or scheduler forwarding authority became public' >&2
    failed=1
  fi
  if ! rg -F -x -q \
      'pub fn TokenizedRequest::new(request_id~ : @inference.RequestId, model_identity~ : @spec.ModelIdentity, input_tokens~ : @inference.TokenBuffer, sampling~ : @inference.SamplingParameters, sampling_seed~ : @inference.SamplingSeed, stop_tokens~ : @inference.LunaRequestStopTokenView, stream_preference~ : @inference.StreamPreference, effective_limits~ : @inference.EffectiveLimits, admission_deadline~ : @inference.AdmissionDeadline) -> Self raise SchedulerError' \
      scheduler/core/pkg.generated.mbti ||
    ! rg -F -x -q \
      'pub fn PreparedAdmission::retain_stop_tokens_into(Self, @inference.LunaRequestStopTokenRetentionSlot) -> Unit raise SchedulerError' \
      scheduler/core/pkg.generated.mbti ||
    ! rg -F -x -q \
      'pub fn PreparedExclusiveAdmission::retain_stop_tokens_into(Self, @inference.LunaRequestStopTokenRetentionSlot) -> Unit raise SchedulerError' \
      scheduler/core/pkg.generated.mbti ||
    rg -n '^pub fn TokenizedRequest::(stops|cache|stop_tokens|semantic)' \
      scheduler/core/pkg.generated.mbti ||
    rg -n 'LunaRequestSemanticView|StopConditions|CachePolicy' \
      scheduler/core/pkg.generated.mbti; then
    printf '%s\n' \
      'scheduler token-only semantic authority surface drifted' >&2
    failed=1
  fi

  if rg -n \
      '@inference\.(LunaRequestSemanticView|StopConditions|CachePolicy)|\.stops\(\)|\.cache\(\)|stop_string_(count|length|byte_at)|cache_scope_(length|byte_at)|token_only\(' \
      scheduler/core --glob '*.mbt' --glob '!**/*_test.mbt' \
      --glob '!**/*_wbtest.mbt' ||
    ! rg -q \
      'priv stop_tokens : @inference\.LunaRequestStopTokenView' \
      scheduler/core/request.mbt ||
    ! rg -q \
      '@inference\.LunaRequestStopTokenRetentionSlot' \
      scheduler/core/types.mbt ||
    ! rg -F -q \
      'retention.is_stop_token(token)' \
      scheduler/core/completion_preflight.mbt; then
    printf '%s\n' \
      'scheduler source escaped the narrow Luna stop-token projection boundary' >&2
    failed=1
  fi
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux service and wire boundaries are valid.'
