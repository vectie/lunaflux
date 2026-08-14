#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run --target native --release tests/device_worker_alloc

generated_c="_build/native/release/build/tests/device_worker_alloc/device_worker_alloc.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'device-worker one-row release C output is missing' >&2
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
  rg "$forbidden" |
    rg -v 'moonbit_malloc.*DeviceWorkerError_2e(InvalidLifecycle|Wire|Executor|CompletionAbortFailed|Invalid)' |
    rg -v 'moonbit_malloc.*WorkerWireError_2e(InvalidFrame|Protocol)' |
    rg -v 'moonbit_malloc.*DeviceStepError_2e(Wire|Device|InvalidPlan|InvalidLifecycle|ExecutorLaunchFailed|InvalidExecutor|ExecutorDevice|ExecutorSampling)' |
    rg -v 'moonbit_malloc.*SamplingError_2e(LimitExceeded|ScratchTooSmall|NonFiniteLogit)' |
    rg -v 'moonbit_malloc.*WorkerProtocolError_2eIdentityOverflow' |
    rg -q .
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
    printf 'device-worker one-row allocation positive control is ineffective: %s\n' \
      "$control" >&2
    exit 1
  fi
done

# The matrix and every plan/frame are built before the measured probe, and one
# exact warm-up call precedes it. Keep this source-order proof next to the
# runtime counter so moving preparation into the window fails visibly.
main_source="tests/device_worker_alloc/main.mbt"
prepare_line="$(rg -n -m1 'let matrix = prepare_cycles' "$main_source" | cut -d: -f1)"
warm_line="$(rg -n -m1 'execute_cycle\(owner, completion_owner, cycles\[0\]' "$main_source" | cut -d: -f1)"
probe_line="$(rg -n 'alloc_probe_begin\(\)' "$main_source" | tail -n1 | cut -d: -f1)"
if [ -z "$prepare_line" ] || [ -z "$warm_line" ] || [ -z "$probe_line" ] ||
  [ "$prepare_line" -ge "$warm_line" ] || [ "$warm_line" -ge "$probe_line" ]; then
  printf '%s\n' 'device-worker one-row preparation/warm/probe order drifted' >&2
  exit 1
fi

# These are the out-of-line definitions reached by the measured canonical
# one-row prefill/final/decode and greedy/stochastic execute lifecycle. Runtime
# interception remains the transitive proof; this list makes compiler-produced
# helper drift fail visibly instead of silently narrowing the corroboration.
for symbol in \
  'device__worker__alloc14execute__cycle(' \
  'device__worker__alloc19authenticate__cycle(' \
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
  'sample__wire__decode(' \
  'select__paged__wire__logit(' \
  'select__paged__scalar__logit(' \
  'sampling31stochastic__sample__at__scalars(' \
  'sampling30prepare__distribution__scalars(' \
  'sampling21prepare__distribution(' \
  'sampling16select__prepared(' \
  'sampling14counter__state(' \
  'sampling13counter__unit(' \
  'decode__paged__bf16__logits(' \
  'read__paged__i32(' \
  'device7Context21copy__to__fixed__host(' \
  'cuda7Context21copy__to__fixed__host(' \
  'append__wire__completion__frame(' \
  'append__prefill__completed(' \
  'append__final__prefill__completed(' \
  'append__decode__completed(' \
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
    printf 'device-worker one-row function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'device-worker one-row path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux device-worker canonical one-row allocation gate passed.'
