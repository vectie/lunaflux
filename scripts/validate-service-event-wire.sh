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


if [ "$failed" -ne 0 ]; then
  exit 1
fi
