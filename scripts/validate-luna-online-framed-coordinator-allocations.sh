#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73
moon test service/online_session \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/online_session/online_session.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna online framed coordinator release C output is missing' >&2
  exit 1
fi

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
  # Only MoonBit's typed exception envelope is language plumbing. No broad
  # Error/Failure-name filter is permitted in this strict path.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

# Startup construction deliberately allocates the aggregate and all fixed
# queue/capability slots. This proves the extractor and predicate can see a
# real allocation before they are used on warmed transport/event paths.
positive_body="$(extract_definition 'new__luna__online__framed__coordinator(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'coordinator allocation positive control is ineffective' >&2
  exit 1
fi

# A MoonBit function with a default argument is emitted as an exact private
# inner C function. Pin both declarations and reject near-name/suffix matches
# so a similarly named helper cannot accidentally satisfy the allocation scan.
ingress_inner_signature='^struct _M0TP[A-Za-z0-9_]* _M0FP46vectie8lunaflux7service15online__session29luna__framed__ingress_2einner[(]$'
ingress_inner_signatures="$(awk -v pattern="$ingress_inner_signature" \
  '$0 ~ pattern { count += 1 } END { print count + 0 }' "$generated_c")"
if [ "$ingress_inner_signatures" != '2' ] ||
  ! rg -q --pcre2 -U \
    'fn luna_framed_ingress\([\s\S]*route_sequence\? : UInt64 = 0UL' \
    service/online_session/coordinator_ingress.mbt; then
  printf '%s\n' \
    'default-argument framed ingress lost its exact generated inner function' >&2
  exit 1
fi
for hostile_signature in \
  'struct _M0FP46vectie8lunaflux7service15online__session29luna__framed__ingress_2einner_suffix(' \
  'struct near_M0FP46vectie8lunaflux7service15online__session29luna__framed__ingress_2einner('; do
  if printf '%s\n' "$hostile_signature" |
    rg -q "$ingress_inner_signature"; then
    printf '%s\n' \
      'framed ingress generated-symbol guard accepted a hostile near name' >&2
    exit 1
  fi
done

# These functions cover accepted ingress, exact tail accounting, framed event
# copy/confirmation, semantic ACK, rejection credit, and disconnect/drain.
# Preparation Ready assembly has one separately documented constant shell and
# is intentionally not misrepresented as a strict zero-allocation transition.
for symbol in \
  'LunaOnlineFramedIngress4kind(' \
  'luna__framed__ingress_2einner(' \
  'luna__framed__ingress__consumed(' \
  'LunaOnlineFramedService27stream__offer__luna__framed(' \
  'LunaOnlineFramedService42stream__offer__luna__framed__with__receipt(' \
  'LunaOnlineFramedService28observe__receipt__completion(' \
  'LunaOnlineFramedService22current__receipt__work(' \
  'LunaOnlineFramedService38stream__copy__framed__event__chunk__to(' \
  'LunaOnlineFramedEventOffer14require__owner(' \
  'LunaOnlineFramedEventOffer6length(' \
  'LunaOnlineFramedEventOffer7confirm(' \
  'LunaOnlineFramedService22start__semantic__event(' \
  'LunaOnlineFramedService25progress__outbound__event(' \
  'LunaOnlineFramedService18finish__event__ack(' \
  'LunaOnlineFramedService18progress__instance(' \
  'LunaOnlineFramedService23progress__reactor__core(' \
  'LunaOnlineFramedService22enforce__output__stall(' \
  'LunaOnlineFramedService14stall__expired(' \
  'LunaOnlineFramedService28release__outbound__authority(' \
  'LunaOnlineFramedService23stream__take__rejection(' \
  'LunaOnlineFramedService29stream__take__semantic__event(' \
  'LunaOnlineFramedService20publish__observation(' \
  'LunaOnlineFramedService27publish__event__observation(' \
  'LunaOnlineFramedService25stream__take__observation(' \
  'LunaOnlineFramedService36publish__worker__failure__if__needed(' \
  'LunaOnlineFramedService22begin__request__timing(' \
  'LunaOnlineFramedService26begin__request__timing__at(' \
  'LunaOnlineFramedService23sample__request__timing(' \
  'LunaOnlineFramedService27sample__request__timing__at(' \
  'LunaOnlineFramedService23retire__request__timing(' \
  'LunaOnlineFramedService22clear__request__timing(' \
  'LunaOnlineFramedObservation14require__owner(' \
  'LunaOnlineFramedObservation21require__usage__owner(' \
  'LunaOnlineFramedObservation4kind(' \
  'LunaOnlineFramedObservation3ack(' \
  'LunaOnlineFramedRejectionCredit14require__owner(' \
  'LunaOnlineFramedRejectionCredit4rule(' \
  'LunaOnlineFramedRejectionCredit3ack(' \
  'LunaOnlineFramedSemanticEvent14require__owner(' \
  'LunaOnlineFramedSemanticEvent4view(' \
  'LunaOnlineFramedSemanticEvent9delivered(' \
  'LunaOnlineFramedService25begin__stream__disconnect(' \
  'LunaOnlineFramedService21begin__service__drain(' \
  'LunaOnlineFramedService27progress__maintenance__core(' \
  'LunaOnlineFramedStream8progress(' \
  'LunaOnlineFramedService23progress__cooperatively('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'coordinator allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'coordinator warmed path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

if ! grep -Fqx \
    'pub fn LunaOnlineFramedStream::offer_luna_framed_with_receipt(Self, @request_admission.LunaRequestReceipt, FixedArray[Byte], source_offset~ : Int, length~ : Int) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  rg -n 'received_at|deadline_at|timestamp|absolute_deadline' \
    service/online_session/framed_stream.mbt \
    service/online_session/coordinator_ingress.mbt |
    rg -v '^[^:]+:[0-9]+:[[:space:]]*///'; then
  printf '%s\n' 'trusted receipt wrapper leaks or rebases receipt authority' >&2
  exit 1
fi

# Pin the integration proof that authenticated coordinator observations cross
# a server-style record-before-ACK retry guard exactly once and that stale
# authority cannot revive when the max-one lane generation is reused.
if ! rg -q -F \
    'test "Luna admitted event sequence records latency once across ACK retry" {' \
    service/online_session/coordinator_observation_wbtest.mbt ||
  ! rg -q --pcre2 -U \
    'record_before_ack[\s\S]*if self\.observation_recorded \{[\s\S]*return[\s\S]*self\.observation_recorded = true' \
    service/online_session/coordinator_observation_wbtest.mbt ||
  ! rg -q --pcre2 -U \
    'server\.first_token_samples, 1[\s\S]*server\.inter_token_samples, 1[\s\S]*server\.request_samples, 1[\s\S]*epoch: 42UL[\s\S]*terminal\.latency_kind\(\)' \
    service/online_session/coordinator_observation_wbtest.mbt; then
  printf '%s\n' \
    'coordinator-to-server latency ACK/reuse regression proof drifted' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'WorkerLatchedFailure =>[\s\S]*LunaOnlineFramedWorkerRequestTerminalObservation' \
    service/online_session/coordinator_observation.mbt ||
  ! rg -q -F \
    'LunaOnlineFramedWorkerRequestTerminalObservation => 10' \
    service/online_session/coordinator_observation.mbt ||
  ! rg -q -F \
    '10 => LunaOnlineFramedWorkerRequestTerminalObservation' \
    service/online_session/coordinator_observation.mbt; then
  printf '%s\n' \
    'worker-loss terminal lost its distinct payload-free observation mapping' >&2
  exit 1
fi

if ! grep -Fqx \
    'pub fn LunaOnlineFramedStream::take_observation(Self) -> LunaOnlineFramedObservation raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedObservation::kind(Self) -> LunaOnlineFramedObservationKind raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedObservation::latency_kind(Self) -> LunaOnlineFramedLatencyKind raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedObservation::latency_millis(Self) -> Int raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedObservation::ack(Self) -> Unit raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! rg -q --pcre2 -U \
    'pub struct LunaOnlineFramedObservation \{\s*// private fields\s*\}' \
    service/online_session/pkg.generated.mbti ||
  ! rg -q 'LunaOnlineFramedCoordinatorObservationReady' \
    service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'finite observation pulse API lost opacity or exact ACK' >&2
  exit 1
fi

if ! grep -Fqx \
    'pub fn LunaOnlineFramedService::open_semantic_stream(Self) -> LunaOnlineFramedOpen' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedStream::take_semantic_event(Self) -> LunaOnlineFramedSemanticEvent raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedSemanticEvent::view(Self) -> @luna_event.LunaEventView raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! grep -Fqx \
    'pub fn LunaOnlineFramedSemanticEvent::delivered(Self) -> Unit raise LunaOnlineFramedCoordinatorError' \
    service/online_session/pkg.generated.mbti ||
  ! rg -q --pcre2 -U \
    'pub struct LunaOnlineFramedSemanticEvent \{\s*// private fields\s*\}' \
    service/online_session/pkg.generated.mbti ||
  ! rg -q 'LunaOnlineFramedCoordinatorSemanticEventReady' \
    service/online_session/pkg.generated.mbti; then
  printf '%s\n' \
    'semantic stream mode lost its opaque exact-delivery API' >&2
  exit 1
fi

# Release compilation inlines these two authenticated scalar projections into
# their E2E callers. Pin their exact source bodies while their transitive
# require_owner helper remains in the strict generated-C set above.
if ! rg -q --pcre2 -U \
    'pub fn LunaOnlineFramedIngress::consumed_bytes[^{]*\{\s*self\.consumed\s*\}' \
    service/online_session/coordinator_ingress.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaOnlineFramedRejectionCredit::sequence[^{]*\{\s*self\.require_owner\(\)\.rejection_sequence\s*\}' \
    service/online_session/coordinator_progress.mbt; then
  printf '%s\n' \
    'inlined coordinator scalar projection stopped being an authenticated field read' >&2
  exit 1
fi

coordinator_fields="$(sed -n \
  '/^pub struct LunaOnlineFramedService {/,/^}/p' \
  service/online_session/coordinator_types.mbt)"
if [ -z "$coordinator_fields" ] ||
  printf '%s\n' "$coordinator_fields" | rg -q '\?' ||
  ! printf '%s\n' "$coordinator_fields" |
    rg -q 'priv work_slots : FixedArray\[' ||
  ! printf '%s\n' "$coordinator_fields" |
    rg -q 'priv stream_slots : Array\[LunaOnlineFramedStream\]' ||
  ! printf '%s\n' "$coordinator_fields" |
    rg -q 'priv timing_slots : FixedArray\[LunaOnlineFramedTimingSlot\]' ||
  ! rg -q 'work_slots: FixedArray::makei\(lane_count, _ => Array::new\(capacity=1\)\)' \
    service/online_session/coordinator_construct.mbt ||
  ! rg -q 'event_credits: Array::new\(capacity=1\)' \
    service/online_session/coordinator_construct.mbt ||
  ! rg -q 'outbound_works: Array::new\(capacity=1\)' \
    service/online_session/coordinator_construct.mbt ||
  ! rg -q 'outbound_views: Array::new\(capacity=1\)' \
    service/online_session/coordinator_construct.mbt ||
  ! rg -q 'stream_slots: Array::new\(capacity=1\)' \
    service/online_session/coordinator_construct.mbt ||
  ! rg -q 'timing_slots: new_luna_online_framed_timing_slots' \
    service/online_session/coordinator_construct.mbt; then
  printf '%s\n' \
    'coordinator warmed owner acquired Option boxing or dynamic authority slots' >&2
  exit 1
fi

if rg -n \
    'GenerateRequest|ReceivedRequest|RequestFrameBuffer|EventFrameBuffer|LunaFramedEventAdapter|pub async fn|Tcp|Socket|Listener' \
    service/online_session/coordinator_types.mbt \
    service/online_session/coordinator_prepare.mbt \
    service/online_session/coordinator_ingress.mbt \
    service/online_session/coordinator_progress.mbt \
    service/online_session/coordinator_events.mbt \
    service/online_session/coordinator_lifecycle.mbt \
    service/online_session/framed_service_owner.mbt \
    service/online_session/framed_stream.mbt \
    service/online_session/framed_semantic_event.mbt \
    service/online_session/coordinator_observation.mbt \
    service/online_session/coordinator_timing.mbt; then
  printf '%s\n' \
    'coordinator reintroduced object materialization, compatibility framing, or network IO' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'try! owner\.outbound_views\[0\]\.release\(\)[\s\S]*owner\.outbound_views\.clear\(\)[\s\S]*owner\.event_ack_pending = true[\s\S]*if owner\.event_kind == 3 \{[\s\S]*owner\.maintenance_kind = 3[\s\S]*LunaOnlineFramedCoordinatorMaintenanceRequired' \
    service/online_session/coordinator_events.mbt ||
  ! rg -q --pcre2 -U \
    'if self\.maintenance_kind == 3 \{[\s\S]*return self\.finish_event_ack\(\)' \
    service/online_session/coordinator_lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'if !self\.tickets\.is_empty\(\) \{[\s\S]*self\.begin_output_disconnect\(\)[\s\S]*\} else \{[\s\S]*self\.release_outbound_authority\(\)' \
    service/online_session/coordinator_lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineFramedService::progress_preparation[\s\S]*if self\.prepared\.length\(\) == 1[\s\S]*return self\.dispatch_prepared\(\)[\s\S]*if self\.queue_count > 0[\s\S]*LunaRequestPreparationReady =>[\s\S]*return self\.dispatch_prepared\(\)[\s\S]*match self\.pool\.progress\(\)' \
    service/online_session/coordinator_progress.mbt ||
  ! rg -q --pcre2 -U \
    'if self\.disconnect_requested \{[\s\S]*if self\.tickets\.is_empty\(\)[\s\S]*return self\.progress_preparation\(\)[\s\S]*return LunaOnlineFramedCoordinatorMaintenanceRequired' \
    service/online_session/coordinator_progress.mbt ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineFramedService::reset_retired_stream[\s\S]*self\.stream_phase = 0[\s\S]*if self\.drain_requested \{[\s\S]*self\.pool\.begin_drain\(\)[\s\S]*self\.instance\.begin_drain\(\)' \
    service/online_session/framed_service_owner.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaOnlineFramedCoordinator::disconnect[\s\S]*self\.stream\.disconnect\(\)[\s\S]*self\.service\.begin_drain\(\)' \
    service/online_session/coordinator_facade.mbt ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineFramedCoordinator::consume_compatibility_observation[\s\S]*take_observation\(\)[\s\S]*observation\.ack\(\)[\s\S]*LunaOnlineFramedCoordinatorAdvanced' \
    service/online_session/coordinator_facade.mbt ||
  rg -q 'self\.observation_kind = 0|self\.observation_taken = false' \
    service/online_session/coordinator_lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'stream_retirement_complete[\s\S]*self\.observation_kind == 0[\s\S]*!self\.observation_taken' \
    service/online_session/framed_service_owner.mbt; then
  printf '%s\n' \
    'coordinator framing, FIFO publication, or disconnect authority order drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux online framed coordinator allocation and authority gate passed.'
