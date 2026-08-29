#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  printf '%s\n' "LunaFlux tensor-collective boundary failed: $1" >&2
  exit 1
}

[ "$(rg -c 'vectie/lunaflux/internal/nccl' device/moon.pkg)" -eq 1 ] ||
  fail 'device must be the exact NCCL facade owner'

if rg -n \
  'model/|scheduler|rank_group|config/|approved_fs|runtime/' \
  device/moon.pkg; then
  fail 'the generic device contract imported higher-level authority'
fi

if rg -n \
  'internal/(nccl|cuda)|scheduler|rank_group|config/|approved_fs|runtime/' \
  internal/tensor_parallel_collective/moon.pkg; then
  fail 'the logical rank-plan verifier acquired execution or root authority'
fi

if matches=$(rg -n 'TensorCollectiveContract::new' --glob '*.mbt' |
  rg -v '^(device/tensor_collective_contract\.mbt|device/.*_wbtest\.mbt|internal/tensor_parallel_collective/authenticate\.mbt):' \
  2>/dev/null); then
  fail "production contract construction escaped the logical verifier: $matches"
fi

if matches=$(rg -n '\.submit_bf16\(' --glob '*.mbt' |
  rg -v '^device/tensor_collective_execute\.mbt:' 2>/dev/null); then
  fail "private NCCL execution escaped the device owner: $matches"
fi

if rg -n \
  'nccl(Get|Comm|Result|Unique)|libnccl|vendor|raw (handle|pointer)|cudaStream_t|CUdeviceptr' \
  device/tensor_collective_*.mbt \
  internal/tensor_parallel_collective --glob '*.mbt' --glob '*.md'; then
  fail 'raw or vendor authority escaped the private native layer'
fi

for claim in claimed_generation claimed_plan_sequence \
  claimed_global_sequence claimed_site_sequence claimed_operation_id \
  claimed_kind live_query_tokens; do
  rg -q "$claim" device/tensor_collective_execute.mbt ||
    fail "scalar launch authentication omitted $claim"
done

if rg -n 'TensorCollectiveLaunchClaim' device \
  internal/tensor_parallel_collective --glob '*.mbt' --glob '*.mbti'; then
  fail 'heap-shaped per-collective launch claim remains reachable'
fi
rg -F -q 'claimed_plan_sequence != current_plan_sequence + 1UL' \
  device/tensor_collective_execute.mbt ||
  fail 'new plans are not bound to the exact predecessor sequence'
rg -F -q 'previous_plan_sequence_value~ : UInt64' \
  device/tensor_collective_runtime.mbt ||
  fail 'communicator owner creation lacks rank-group predecessor authority'
rg -F -q 'next_site_index: contract.site_count()' \
  device/tensor_collective_runtime.mbt ||
  fail 'first plan does not begin after an authenticated predecessor'
rg -F -q 'live_query_tokens > contract.max_query_tokens' \
  device/tensor_collective_execute.mbt ||
  fail 'submit does not authenticate the startup query-token envelope'
rg -F -q 'checked_tensor_collective_elements(' \
  device/tensor_collective_execute.mbt ||
  fail 'submit does not use checked Int64 element multiplication'
rg -F -q 'encoder.append_int(max_query_tokens)' \
  internal/tensor_parallel_collective/authenticate.mbt ||
  fail 'canonical contract digest omits the maximum query-token claim'
rg -F -q 'pub fn TensorCollectiveContract::max_query_tokens(' \
  device/tensor_collective_contract.mbt ||
  fail 'startup admission cannot inspect the authenticated token envelope'

rg -F -q 'self.communicator.invalidate(generation=self.generation)' \
  device/tensor_collective_execute.mbt ||
  fail 'authentication substitution does not invalidate owned generation'
rg -q 'self\.state = CollectiveOwnerFailed' \
  device/tensor_collective_execute.mbt ||
  fail 'collective failures are not latched'
rg -q 'MINIMUM_NCCL_BF16_VERSION_CODE : Int = 21403' \
  device/tensor_collective_runtime.mbt ||
  fail 'runtime admission no longer requires nonblocking NCCL 2.14.3 or newer'
rg -F -q 'context=self.context.raw' device/tensor_collective_execute.mbt ||
  fail 'execution does not revalidate resources against the owner context'
if rg -n 'context~ : Context' device/tensor_collective_execute.mbt; then
  fail 'execute accepts redundant caller context authority'
fi

if rg -n 'TO''DO|H''ACK|FIX''ME' device/tensor_collective_*.mbt \
  internal/tensor_parallel_collective internal/nccl/collectives.c; then
  fail 'temporary marker remains in the collective bridge'
fi

for source_file in device/tensor_collective_*.mbt \
  internal/tensor_parallel_collective/* internal/nccl/collectives.c \
  internal/nccl/collective_probe.c; do
  [ -f "$source_file" ] || continue
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  [ "$line_count" -le 500 ] ||
    fail "$source_file exceeds the 500-line boundary budget"
done

printf '%s\n' 'LunaFlux authenticated tensor-collective boundary is valid.'
