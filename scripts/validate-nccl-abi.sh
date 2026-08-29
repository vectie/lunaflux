#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  printf '%s\n' "LunaFlux NCCL ABI gate failed: $1" >&2
  exit 1
}

if matches=$(rg -n \
  'vectie/lunaflux/internal/nccl' --glob 'moon.pkg' |
  rg -v '^device/moon\.pkg:' 2>/dev/null); then
  fail "only the device ownership boundary may import internal/nccl: $matches"
fi

outside_nccl_files=$(rg --files --glob '*.mbt' --glob '*.c' --glob '*.h' |
  sed '/^internal\/nccl\//d')
if [ -n "$outside_nccl_files" ] &&
  matches=$(printf '%s\n' "$outside_nccl_files" |
    xargs rg -n \
      'nccl(Get|Comm|Result|Unique)|ncclResult_t|ncclComm_t|lf_nccl_|libnccl\.so' \
      2>/dev/null); then
  fail "raw NCCL vocabulary escaped internal/nccl: $matches"
fi

if rg -n 'TO''DO|H''ACK|FIX''ME' internal/nccl 2>/dev/null; then
  fail 'temporary marker remains in internal/nccl'
fi

if rg -n \
  'pub extern "c"|ncclGet(ErrorString|LastError)|pub (struct|enum|type|fn).*(Vendor|Topology|Model|Scheduler|Handle)' \
  internal/nccl --glob '*.mbt' 2>/dev/null; then
  fail 'raw/vendor/product vocabulary escaped the private Moon facade'
fi

if rg -n 'cc-link-flags|-lnccl|-lcuda|-lcudart' internal/nccl/moon.pkg; then
  fail 'internal/nccl must have no production static NCCL or CUDA dependency'
fi

rg -q 'vectie/lunaflux/internal/device_collective_interop' \
  internal/nccl/moon.pkg || fail 'NCCL does not import neutral device tokens'
if rg -n '@cuda' internal/nccl internal/device_collective_interop \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
  2>/dev/null; then
  fail 'NCCL or neutral token APIs expose CUDA'
fi
if [ "$(rg -l '@cuda.Context' internal/nccl --glob '*_test.mbt' \
  --glob '*_wbtest.mbt' | wc -l | tr -d ' ')" -ne 2 ]; then
  fail 'CUDA test-link evidence escaped its two focused test modes'
fi
if [ "$(rg -c 'vectie/lunaflux/internal/cuda' internal/nccl/moon.pkg)" -ne 2 ] ||
  ! rg -q -U \
  'import \{\n  "vectie/lunaflux/internal/cuda",\n\} for "test"\n\nimport \{\n  "vectie/lunaflux/internal/cuda",\n\} for "wbtest"' \
  internal/nccl/moon.pkg; then
  fail 'CUDA linking must remain confined to exact test and wbtest clauses'
fi
if rg -n 'vectie/lunaflux/internal/cuda' \
  internal/device_collective_interop/moon.pkg 2>/dev/null; then
  fail 'neutral token ownership depends on CUDA'
fi
if matches=$(rg -n \
  'resource_internal|lf_context|lf_allocation|lf_child|CU(device|context|stream|result)|cu[A-Z]|CUDA_(SUCCESS|ERROR)' \
  internal/nccl --glob '*.c' --glob '*.h' 2>/dev/null); then
  fail "raw device resource vocabulary escaped through NCCL: $matches"
fi
matches=
if [ "$(rg -c '#include "\.\./cuda/collective_interop\.h"' \
  internal/nccl/nccl_abi.h)" -ne 1 ] ||
  matches=$(rg -n '#include "\.\./cuda/' internal/nccl \
    --glob '*.c' --glob '*.h' |
    rg -v 'nccl_abi\.h:.*collective_interop\.h|_probe\.c:.*collective_interop_probe\.h' \
    2>/dev/null); then
  fail "NCCL crossed the single production interop header: $matches"
fi
if rg -n 'moonbit_(incref|decref)' \
  internal/nccl/communicator.c internal/nccl/collectives.c 2>/dev/null; then
  fail 'NCCL directly manipulates device-owner references'
fi

if [ "$(rg -c 'dlopen\("libnccl\.so\.2"' internal/nccl/loader.c)" -ne 1 ] ||
  rg -n 'dlopen\("libnccl\.so"' internal/nccl/loader.c; then
  fail 'loader must admit only the NCCL v2 SONAME'
fi

for symbol in ncclGetVersion ncclGetUniqueId ncclCommInitRankConfig \
  ncclCommGetAsyncError ncclCommDestroy ncclCommAbort ncclAllReduce \
  ncclAllGather; do
  [ "$(rg -c "\"$symbol\"" internal/nccl/loader.c)" -eq 1 ] ||
    fail "required runtime symbol admission drifted: $symbol"
done

for native_stub in collective_probe.c collectives.c communicator.c \
  communicator_ffi.c lifecycle_probe.c loader.c; do
  rg -q "\"$native_stub\"" internal/nccl/moon.pkg ||
    fail "native-stub list is missing $native_stub"
done

rg -q '"stub-cc-flags": "-std=c11 -Wall -Wextra -Werror"' \
  internal/nccl/moon.pkg || fail 'strict C11 compilation flags are missing'

rg -q -U \
  '#borrow\(unique_id, status\)\nextern "c" fn raw_communicator_create' \
  internal/nccl/ffi.mbt || fail 'communicator creation must borrow ID and status'
rg -q -U \
  '#borrow\(communicator\)\nextern "c" fn raw_communicator_close' \
  internal/nccl/ffi.mbt || fail 'communicator close must borrow its owner'
rg -q -U \
  '#borrow\(communicator\)\nextern "c" fn raw_communicator_abort' \
  internal/nccl/ffi.mbt || fail 'communicator abort must borrow its owner'
rg -q -U \
  '#borrow\(communicator, context, send, receive, stream\)\nextern "c" fn raw_communicator_submit_bf16' \
  internal/nccl/ffi.mbt || fail 'collective submit must borrow every native owner'
rg -q -U \
  '#borrow\(communicator\)\nextern "c" fn raw_communicator_poll_collective_state' \
  internal/nccl/ffi.mbt || fail 'collective poll must borrow only its owner'
if rg -n 'raw_communicator_poll_collective\(' internal/nccl/ffi.mbt \
  internal/nccl/api.mbt ||
  ! rg -q 'int32_t completed = 0;' internal/nccl/collectives.c ||
  ! rg -F -q 'return status == LF_NCCL_OK ? completed : status;' \
    internal/nccl/collectives.c ||
  [ "$(rg -c 'lunaflux_nccl_communicator_poll_collective_state' \
    internal/nccl/nonblocking_sanitizer_probe.c)" -ne 4 ]; then
  fail 'collective poll state ABI lost its stack-only scalar result'
fi
rg -q -U \
  '#borrow\(unique_id, context, status\)\nextern "c" fn raw_communicator_create_on_context' \
  internal/nccl/ffi.mbt || fail 'communicator creation must borrow its exact CUDA context'

rg -F -q 'lf_device_context_lease_acquire(' internal/nccl/communicator.c ||
  fail 'context-bound communicator does not acquire the opaque context lease'
rg -F -q 'lf_device_context_lease_release(&communicator->context_lease);' \
  internal/nccl/communicator.c ||
  fail 'communicator consumption does not release context authority'
rg -q 'LF_NCCL_NONBLOCKING_MINIMUM_VERSION 21403' internal/nccl/nccl_abi.h ||
  fail 'runtime admission no longer requires NCCL 2.14.3'
rg -q 'LF_NCCL_CONFIG_VERSION 21403U' internal/nccl/nccl_abi.h ||
  fail 'the conservative v2.14.3 configuration version drifted'
rg -F -q '.blocking = 0' internal/nccl/communicator.c ||
  fail 'communicator initialization is no longer explicitly nonblocking'
rg -F -q 'sizeof(lf_nccl_config_v21400) == 24' \
  internal/nccl/communicator.c ||
  fail 'the admitted v2.14 configuration layout is not statically checked'

rg -q 'next_collective_sequence = 1' internal/nccl/communicator.c ||
  fail 'collective sequence must start at one'
rg -q 'collective_sequence == 0' internal/nccl/communicator.c ||
  fail 'zero collective sequence must fail closed'
rg -q 'last_operation_id = -1' internal/nccl/communicator.c ||
  fail 'within-plan operation ordering lacks a safe sentinel'
rg -q 'operation_id <= communicator->last_operation_id' \
  internal/nccl/communicator.c ||
  fail 'same-plan operation replay/substitution is not rejected'
rg -q 'world_size < LF_NCCL_MIN_WORLD_SIZE' internal/nccl/communicator.c ||
  fail 'communicator world-size lower bound is missing'
rg -q 'world_size > LF_NCCL_MAX_WORLD_SIZE' internal/nccl/communicator.c ||
  fail 'communicator world-size upper bound is missing'
rg -F -q 'atomic_store(&communicator->failed, 1)' \
  internal/nccl/communicator.c || fail 'generation/sequence failure latch is missing'
rg -F -q 'plan_sequence != communicator->last_plan_sequence + 1' \
  internal/nccl/communicator.c ||
  fail 'new plan sequence is not the exact predecessor plus one'
rg -q 'lf_nccl_collective_begin' internal/nccl/collectives.c ||
  fail 'collective execution does not begin scalar authentication atomically'
if rg -n 'Synchronize|synchronize' internal/nccl --glob '*.c'; then
  fail 'blocking device synchronization remains in the NCCL path'
fi
rg -q 'comm_get_async_error' internal/nccl/collectives.c ||
  fail 'collective poll does not authenticate NCCL async state'
rg -q 'lf_device_collective_lease_query' internal/nccl/collectives.c ||
  fail 'collective poll does not query opaque device completion'
rg -q 'LF_NCCL_PHASE_IN_FLIGHT' internal/nccl/collectives.c ||
  fail 'fixed in-flight state is missing'
rg -q 'LF_NCCL_PHASE_FAILED' internal/nccl/collectives.c ||
  fail 'failed asynchronous state is not retained'
rg -q 'lf_nccl_release_in_flight' internal/nccl/collectives.c ||
  fail 'in-flight device lease lacks a consuming release path'
rg -q 'lf_device_collective_lease_acquire' internal/nccl/collectives.c ||
  fail 'collective submit does not atomically acquire exact device resources'
rg -q 'lf_device_collective_lease_view' internal/nccl/collectives.c ||
  fail 'NCCL enqueue lacks the narrow vendor-neutral token view'
rg -q 'lf_nccl_collective_commit' internal/nccl/collectives.c ||
  fail 'completed collective poll does not commit its sequence'
rg -q 'NCCL_BFLOAT16 9' internal/nccl/collectives.c ||
  fail 'the admitted NCCL BF16 ABI value drifted'

for source_file in $(rg --files internal/nccl scripts \
  --glob '*.mbt' --glob '*.c' --glob '*.h' |
  rg '^(internal/nccl/|scripts/validate-nccl-)'); do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  [ "$line_count" -le 500 ] ||
    fail "$source_file exceeds the 500-line native boundary budget"
done

printf '%s\n' 'LunaFlux NCCL ABI and ownership boundary is valid.'
