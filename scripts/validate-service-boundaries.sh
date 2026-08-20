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
    'framed service wire imports outside contracts/inference + model/spec:' \
    --pcre2 --glob 'service/framed_wire/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|model/spec")'
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
      '^pub fn admit\(ReceivedRequest, @tokenizer\.TokenizerSpec, @tokenizer\.TokenizerDigest, @spec\.ModelIdentity, @inference\.InferenceLimits, @monotonic_clock\.MonotonicClock\)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'request admission must retain typed receipt/model/tokenizer binding' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct (IncrementalRequestReceiver|ReceivedRequest|AdmittedRequest) \{\n  (?!// private fields)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' 'request admission owners must remain opaque' >&2
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
