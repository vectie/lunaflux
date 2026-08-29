#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run --target native --release tests/device_worker_alloc

generated_c="_build/native/release/build/tests/device_worker_alloc/device_worker_alloc.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'device-worker batch release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|double|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ { candidate = 1; body = $0 ORS; next }
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
    rg -v 'moonbit_malloc.*device_2eDeviceError_2e(InvalidArgument|Unavailable|Closed|Busy|DriverFailure|HostAllocationFailed|SizeOverflow|Unsupported|InvalidOutput)' |
    rg -v 'moonbit_malloc.*cuda_2eCudaError_2e(InvalidArgument|Unavailable|Closed|Busy|DriverFailure|HostAllocationFailed|SizeOverflow|Unsupported|InvalidOutput)' |
    rg -v 'moonbit_malloc.*SamplingError_2e(LimitExceeded|ScratchTooSmall|NonFiniteLogit)' |
    rg -v 'moonbit_malloc.*WorkerProtocolError_2eIdentityOverflow' |
    rg -q .
}

for control in \
  'positive__record__control(' \
  'positive__array__control(' \
  'positive__string__control('; do
  body="$(extract_definition "$control")"
  if [ -z "$body" ] || ! printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'device-worker batch allocation positive control is ineffective: %s\n' \
      "$control" >&2
    exit 1
  fi
done

# Fail closed if the production stochastic path stops reaching the bounded
# top-k selector or if its heap/sort stages disappear from generated release C.
# The executable's authenticated mixed-batch fixture keeps this call chain live;
# every body is also checked below for allocation sites.
sampling_prepare_body="$(extract_definition 'prepare__distribution__scalars(')"
candidate_order_body="$(extract_definition 'prepare__candidate__order(')"
top_k_selector_body="$(extract_definition 'select__top__k__candidates(')"
if [ -z "$sampling_prepare_body" ] ||
  ! printf '%s\n' "$sampling_prepare_body" |
    rg -F -q 'prepare__candidate__order('; then
  printf '%s\n' 'sampling preparation lost the retained candidate-order call' >&2
  exit 1
fi
if [ -z "$candidate_order_body" ] ||
  ! printf '%s\n' "$candidate_order_body" |
    rg -F -q 'select__top__k__candidates(' ||
  ! printf '%s\n' "$candidate_order_body" |
    rg -F -q 'sort__candidates('; then
  printf '%s\n' 'sampling candidate-order branch structure drifted' >&2
  exit 1
fi
if [ -z "$top_k_selector_body" ] ||
  ! printf '%s\n' "$top_k_selector_body" | rg -F -q 'make__worst__heap(' ||
  ! printf '%s\n' "$top_k_selector_body" | rg -F -q 'sift__worst__down(' ||
  ! printf '%s\n' "$top_k_selector_body" | rg -F -q 'drain__worst__heap('; then
  printf '%s\n' 'sampling top-k retained-heap call structure drifted' >&2
  exit 1
fi

main_source="tests/device_worker_alloc/main.mbt"
prepare_line="$(rg -n -m1 'let matrix = prepare_batch_matrix' "$main_source" | cut -d: -f1)"
warm_line="$(rg -n -m1 'execute_batch_cycle\(owner, completion_owner, cycles\[0\]' "$main_source" | cut -d: -f1)"
probe_line="$(rg -n 'alloc_probe_begin\(\)' "$main_source" | tail -n1 | cut -d: -f1)"
if [ -z "$prepare_line" ] || [ -z "$warm_line" ] || [ -z "$probe_line" ] ||
  [ "$prepare_line" -ge "$warm_line" ] || [ "$warm_line" -ge "$probe_line" ]; then
  printf '%s\n' 'device-worker batch preparation/warm/probe order drifted' >&2
  exit 1
fi

for symbol in \
  'device__worker__alloc21execute__batch__cycle(' \
  'device__worker__alloc26authenticate__batch__cycle(' \
  'device__worker__alloc28authenticate__batch__prefill(' \
  'device__worker__alloc31authenticate__batch__completion(' \
  'DeviceWorkerOwner14execute__frame(' \
  'device__worker15ready__executor(' \
  'CompletionFrameBuffer11writer__for(' \
  'PagedGraphExecutor12stage__frame(' \
  'DeviceStepOwner34stage__frame__with__sampling__mode(' \
  'DeviceStepOwner35preflight__validated__frame_2einner(' \
  'preflight__frame__and__write_2einner(' \
  'DeviceStepOwner25stage__preflighted__frame(' \
  'device__step26upload__owner__transaction(' \
  'write__frame__prefill(' \
  'write__frame__decode(' \
  'device__step13copy__operand(' \
  'PagedGraphExecutor7execute(' \
  'authenticate__staged__paged(' \
  'device21OrderedKernelExecutor7enqueue(' \
  'cuda21OrderedKernelExecutor7enqueue(' \
  'device21OrderedKernelExecutor18record__completion(' \
  'cuda21OrderedKernelExecutor18record__completion(' \
  'device21OrderedKernelExecutor16wait__completion(' \
  'cuda21OrderedKernelExecutor16wait__completion(' \
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
  'prepare__candidate__order(' \
  'select__top__k__candidates(' \
  'make__worst__heap(' \
  'drain__worst__heap(' \
  'sampling16sort__candidates(' \
  'sampling17sift__worst__down(' \
  'preferred__before(' \
  'sampling11worse__than(' \
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
  'device21OrderedKernelExecutor5reset(' \
  'cuda21OrderedKernelExecutor5reset(' \
  'DeviceStepOwner6finish(' \
  'CompletionFrameWriter6submit(' \
  'completion__frame__checksum(' \
  'ValidatedCompletionFrame13validate__for(' \
  'ValidatedCompletionFrame16entry__for__slot(' \
  'SchedulePlanBuffer6retire(' \
  'SchedulePlanBuffer5reset('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'device-worker batch function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'device-worker batch path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

if ! rg -Fq '"fake_kernel.c"' tests/device_worker_alloc/moon.pkg ||
  rg -n 'native-stub.*\.\.|"\.\./.*\.c"' tests/device_worker_alloc/moon.pkg; then
  printf '%s\n' 'device-worker allocation harness escaped its focused stub root' >&2
  exit 1
fi

for file in tests/device_worker_alloc/*; do
  lines="$(wc -l < "$file")"
  if [ "$lines" -ge 500 ]; then
    printf '%s exceeds the strict 499-line allocation harness budget\n' "$file" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux device-worker mixed/full-batch allocation gate passed.'
