#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

fail() {
  printf '%s\n' "FP8 v3 frame allocation gate: $1" >&2
  exit 1
}

test_source=engine/device_step/fp8_frame_admission_alloc_wbtest.mbt
admission_source=engine/device_step/fp8_frame_admission.mbt
run_source=engine/device_step/paged_executor_run.mbt

moon test --target native --release --deny-warn --warn-list +73 \
  "$test_source"
moon build --target native --release tests/device_worker_alloc

test_generated=_build/native/release/test/engine/device_step/device_step.whitebox_test.c
production_generated=_build/native/release/build/tests/device_worker_alloc/device_worker_alloc.c
[ -f "$test_generated" ] || fail 'release-generated white-box C is missing'
[ -f "$production_generated" ] || fail 'release-generated production C is missing'

extract_function() {
  generated=$1
  symbol=$2
  awk -v symbol="$symbol" '
    !active && !candidate && index($0, symbol) > 0 && $0 ~ /\($/ {
      candidate = 1
      body = $0 ORS
      next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\) \{$/) {
        printf "%s", body
        candidate = 0
        active = 1
        next
      }
      if ($0 ~ /^\);$/) {
        candidate = 0
        body = ""
      }
      next
    }
    active {
      print
      if ($0 ~ /^}$/) exit
    }
  ' "$generated"
}

allocation_pattern='moonbit_malloc|moonbit_make_|moonbit_add_string|moonbit_unsafe_bytes_sub_string'
positive=$(extract_function \
  "$test_generated" 'fp8__admission__allocation__positive__control')
warmed=$(extract_function \
  "$test_generated" 'fp8__admission__warmed__cycles')
publish=$(extract_function \
  "$production_generated" '30publish__fp8__frame__token__v3')
authenticated_publish=$(extract_function \
  "$production_generated" '49authenticate__and__publish__fp8__frame__token__v3')
arm=$(extract_function \
  "$production_generated" '30arm__fp8__frame__admission__v3')
consume=$(extract_function \
  "$production_generated" '30consume__fp8__frame__token__v3')
authenticated_consume=$(extract_function \
  "$production_generated" '49authenticate__and__consume__fp8__frame__token__v3')
clear=$(extract_function \
  "$production_generated" '32clear__fp8__frame__admission__v3')
clear_state=$(extract_function \
  "$production_generated" '39clear__fp8__frame__admission__v3__state')

[ -n "$positive" ] || fail 'compiled allocation positive control is missing'
[ -n "$warmed" ] || fail 'compiled warmed admission cycle is missing'
[ -n "$publish" ] || fail 'compiled production publication edge is missing'
for body in \
  "$authenticated_publish" "$arm" "$authenticated_consume" \
  "$consume" "$clear" "$clear_state"; do
  [ -n "$body" ] || fail 'compiled transitive publication/consume edge is missing'
done

printf '%s\n' "$positive" | rg -q "$allocation_pattern" || \
  fail 'compiled allocation positive control does not allocate'
if printf '%s\n' "$warmed" | rg -q "$allocation_pattern"; then
  fail 'compiled warmed success/clear/rearm cycle allocates'
fi
for edge in \
  'publish__fp8__frame__token__v3' \
  'consume__fp8__frame__token__v3'; do
  printf '%s\n' "$warmed" | rg -q "$edge" || \
    fail "compiled warmed cycle lost $edge"
done

check_only_error_allocations() {
  label=$1
  body=$2
  allocations=$(printf '%s\n' "$body" | rg "$allocation_pattern" || true)
  if [ -n "$allocations" ] && \
    printf '%s\n' "$allocations" | rg -v -q 'Error|error'; then
    fail "$label contains a non-error allocation site"
  fi
}

check_only_error_allocations authenticated-publish "$authenticated_publish"
check_only_error_allocations publish "$publish"
check_only_error_allocations arm "$arm"
check_only_error_allocations authenticated-consume "$authenticated_consume"
check_only_error_allocations consume "$consume"
check_only_error_allocations clear "$clear"
check_only_error_allocations clear-state "$clear_state"

printf '%s\n' "$authenticated_publish" | \
  rg -q '30publish__fp8__frame__token__v3' || \
  fail 'authenticated publication no longer reaches scalar publication'
printf '%s\n' "$publish" | rg -q '30arm__fp8__frame__admission__v3' || \
  fail 'scalar publication no longer reaches admission arm'
printf '%s\n' "$authenticated_consume" | \
  rg -q '30consume__fp8__frame__token__v3' || \
  fail 'authenticated consume no longer reaches scalar consume'
printf '%s\n' "$consume" | rg -q '32clear__fp8__frame__admission__v3' || \
  fail 'scalar consume no longer retires executor admission'
printf '%s\n' "$clear" | \
  rg -q '39clear__fp8__frame__admission__v3__state' || \
  fail 'executor clear no longer reaches scalar-state clear'

if printf '%s\n' "$publish" | rg -q \
  'Fp8FrameEvidenceV3|Fp8AdmittedFrameV3|Option.*Some'; then
  fail 'production publication reconstructs boxed per-frame evidence'
fi

for removed in Fp8FrameEvidenceV3 Fp8AdmittedFrameV3 pending_fp8_frame; do
  if rg -q "$removed" engine/device_step; then
    fail "retired per-frame owner remains: $removed"
  fi
done

for invariant in \
  'authenticate_and_publish_fp8_frame_token_v3(' \
  'authenticate_and_consume_fp8_frame_token_v3(' \
  'clear_fp8_frame_admission_v3(executor)' \
  'descriptor.stage_preflighted_frame(frame, summary)'; do
  rg -F -q "$invariant" "$admission_source" "$run_source" || \
    fail "missing lifecycle invariant: $invariant"
done

preflight_calls=$(sed -n \
  '/^fn preflight_fp8_frame_v3(/,/^}/p' "$admission_source" | \
  rg -c 'preflight_validated_frame\(frame\)')
[ "$preflight_calls" -eq 1 ] || \
  fail "FP8 read-only preflight count drifted: $preflight_calls"
if sed -n '/^fn preflight_fp8_frame_v3(/,/^}/p' "$admission_source" | \
  rg -q 'prefill_row|decode_row|sequence_token_start|sequence_token_position'; then
  fail 'FP8 route restored a redundant post-preflight row traversal'
fi
if sed -n '/Some(envelope) => {/,/^[[:space:]]*None => {/p' "$run_source" | \
  rg -q 'descriptor\.stage_frame\(frame\)'; then
  fail 'FP8 route restored the redundant descriptor preflight scan'
fi

printf '%s\n' \
  'LunaFlux FP8 v3 warmed frame allocation and two-scan admission gate passed.'
