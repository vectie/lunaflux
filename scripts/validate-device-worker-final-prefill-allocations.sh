#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run --target native --release tests/device_worker_alloc

generated_c="_build/native/release/build/tests/device_worker_alloc/device_worker_alloc.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'device-worker final-prefill release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|double|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
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

# The same runtime counter used for the target proves direct-record,
# FixedArray, and dynamic-string controls independently by category. This
# generated body is a positive control for the exact release-C predicate too.
for control in \
  'positive__record__control(' \
  'positive__array__control(' \
  'positive__string__control('; do
  positive_body="$(extract_definition "$control")"
  if [ -z "$positive_body" ] ||
    ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
    printf 'device-worker final-prefill allocation positive control is ineffective: %s\n' \
      "$control" >&2
    exit 1
  fi
done

# These are the out-of-line definitions reached by the measured final-prefill
# greedy execute lifecycle. Runtime interception remains the transitive proof;
# this list makes compiler-produced helper drift fail visibly instead of
# silently narrowing the static corroboration.
for symbol in \
  'DeviceWorkerOwner14execute__frame(' \
  'device__worker15ready__executor(' \
  'CompletionFrameBuffer11writer__for(' \
  'PagedGraphExecutor12stage__frame(' \
  'DeviceStepOwner12stage__frame(' \
  'preflight__frame__and__write(' \
  'device__step13copy__operand(' \
  'PagedGraphExecutor7execute(' \
  'authenticate__staged__paged(' \
  'device8Function19launch__synchronous(' \
  'cuda8Function19launch__synchronous(' \
  'PagedGraphExecutor25sample__completion__frame(' \
  'authenticate__executed__paged(' \
  'validate__wire__completion__plan(' \
  'CompletionFrameWriter14validate__plan(' \
  'sample__wire__prefill(' \
  'select__paged__wire__logit(' \
  'select__paged__scalar__logit(' \
  'decode__paged__bf16__logits(' \
  'read__paged__i32(' \
  'device7Context21copy__to__fixed__host(' \
  'cuda7Context21copy__to__fixed__host(' \
  'append__wire__completion__frame(' \
  'append__final__prefill__completed(' \
  'write__direct__completion__entry(' \
  'PagedGraphExecutor6finish(' \
  'DeviceStepOwner6finish(' \
  'CompletionFrameWriter6submit(' \
  'completion__frame__checksum(' \
  'ValidatedCompletionFrame13validate__for(' \
  'SchedulePlanBuffer6retire(' \
  'SchedulePlanBuffer5reset('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'device-worker final-prefill function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'device-worker final-prefill path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux device-worker final-prefill allocation gate passed.'
