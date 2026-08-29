#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

if matches=$(rg -n \
  --glob 'moon.pkg' \
  --glob '!device/moon.pkg' \
  --glob '!internal/cuda/moon.pkg' \
  --glob '!internal/nccl/moon.pkg' \
  --glob '!tests/physical_cuda/moon.pkg' \
  'vectie/lunaflux/internal/cuda' 2>/dev/null); then
  printf '%s\n%s\n' \
    'only device may import the private CUDA package:' \
    "$matches" >&2
  failed=1
fi

if [ "$(rg -c 'vectie/lunaflux/internal/cuda' internal/nccl/moon.pkg)" -ne 2 ] ||
  ! rg -q -U \
  'import \{\n  "vectie/lunaflux/internal/cuda",\n\} for "test"\n\nimport \{\n  "vectie/lunaflux/internal/cuda",\n\} for "wbtest"' \
  internal/nccl/moon.pkg; then
  printf '%s\n' \
    'NCCL may link CUDA only in exact test and wbtest clauses' >&2
  failed=1
fi

outside_cuda_files=$(rg --files \
  --glob '*.mbt' --glob '*.c' --glob '*.h' |
  sed '/^internal\/cuda\//d; /^tests\/physical_cuda\//d')
if [ -n "$outside_cuda_files" ] && matches=$(printf '%s\n' "$outside_cuda_files" |
  xargs rg -n \
    'CU(device|context|stream|event|module|function|result)|cu[A-Z]|cublasLt|CUDA_SUCCESS' \
    2>/dev/null); then
  printf '%s\n%s\n' 'raw CUDA vocabulary escaped internal/cuda:' "$matches" >&2
  failed=1
fi

if matches=$(rg -n \
  --glob 'internal/cuda/**' \
  --glob 'device/**' \
  'TO''DO|H''ACK|FIX''ME' 2>/dev/null); then
  printf '%s\n%s\n' 'temporary marker in native device boundary:' "$matches" >&2
  failed=1
fi

while IFS= read -r source_file; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; native boundary files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    failed=1
  fi
done <<EOF
$(rg --files internal/cuda device --glob '*.mbt' --glob '*.c' --glob '*.h' | sort)
EOF

for native_stub in allocation_lease.c allocation_lease_probe.c cublas.c collective_interop.c gemm.c launch.c loader.c modules.c ordered_executor.c ordered_executor_probe.c ordered_graph.c ordered_graph_probe.c peer_access.c peer_access_probe.c region_probe.c regions.c resources.c transfer_probe.c transfers.c; do
  if ! rg -q "\"$native_stub\"" internal/cuda/moon.pkg; then
    printf '%s: %s\n' \
      'internal/cuda native-stub list is missing' "$native_stub" >&2
    failed=1
  fi
done

for graph_symbol in cuStreamBeginCapture cuStreamEndCapture cuGraphInstantiateWithFlags cuGraphDestroy cuGraphExecDestroy cuGraphLaunch; do
  if ! rg -Fq "lf_symbol(api->driver_library, \"$graph_symbol\")" \
    internal/cuda/loader.c; then
    printf '%s: %s\n' 'optional CUDA graph ABI symbol is missing' \
      "$graph_symbol" >&2
    failed=1
  fi
done

if rg -n 'vectie/lunaflux/internal/cuda' \
  internal/device_collective_interop/moon.pkg 2>/dev/null; then
  printf '%s\n' \
    'neutral collective token ownership must not depend on CUDA' >&2
  failed=1
fi

if ! rg -q 'vectie/lunaflux/internal/device_collective_interop' \
  internal/cuda/moon.pkg; then
  printf '%s\n' 'CUDA must return neutral collective token types' >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(context\)\nextern "c" fn raw_collective_context_token' \
  internal/cuda/ffi.mbt ||
  ! rg -q -U \
  '#borrow\(region\)\nextern "c" fn raw_collective_region_token' \
  internal/cuda/ffi.mbt ||
  ! rg -q -U \
  '#borrow\(queue\)\nextern "c" fn raw_collective_queue_token' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' 'collective token conversions must borrow exact owners' >&2
  failed=1
fi

interop_hot_path=$(sed -n \
  '/^int32_t lf_device_collective_lease_acquire(/,/^}/p; /^int32_t lf_device_collective_lease_query(/,/^}/p; /^void lf_device_collective_lease_release(/,/^}/p' \
  internal/cuda/collective_interop.c)
if [ -z "$interop_hot_path" ]; then
  printf '%s\n' 'device collective interop lease implementation is missing' >&2
  failed=1
elif matches=$(printf '%s\n' "$interop_hot_path" | rg -n \
  'moonbit_make|malloc|calloc|realloc|free' 2>/dev/null); then
  printf '%s\n%s\n' \
    'device collective interop hot path allocated:' "$matches" >&2
  failed=1
fi

if ! rg -Fq 'LF_LOAD_REQUIRED(cuDeviceCanAccessPeer, "cuDeviceCanAccessPeer")' \
  internal/cuda/loader.c ||
  ! rg -Fq 'CUresult (*cuDeviceCanAccessPeer)(int32_t *, CUdevice, CUdevice);' \
    internal/cuda/cuda_abi.h; then
  printf '%s\n' 'directed CUDA peer query is not dynamically admitted' >&2
  failed=1
fi

if ! rg -Fq 'LF_LOAD_REQUIRED(cuEventQuery, "cuEventQuery")' \
  internal/cuda/loader.c ||
  ! rg -Fq 'CUresult (*cuEventQuery)(CUevent);' internal/cuda/cuda_abi.h; then
  printf '%s\n' 'ordered completion event query is not dynamically admitted' >&2
  failed=1
fi

if ! rg -Fq 'LF_LOAD_REQUIRED(cuEventSynchronize, "cuEventSynchronize")' \
  internal/cuda/loader.c ||
  ! rg -Fq 'CUresult (*cuEventSynchronize)(CUevent);' \
    internal/cuda/cuda_abi.h; then
  printf '%s\n' 'ordered completion event wait is not dynamically admitted' >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(context, stream, functions, dimensions, argument_starts, allocations, offsets, byte_counts, alignments, status\)\nextern "c" fn raw_ordered_executor_create' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' \
    'ordered executor creation must borrow every startup input' >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(executor\)\nextern "c" fn raw_ordered_executor_wait' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' 'ordered completion wait must borrow its exact owner' >&2
  failed=1
fi

ordered_wait_body=$(sed -n \
  '/^int32_t lunaflux_cuda_ordered_executor_wait(/,/^}/p' \
  internal/cuda/ordered_executor.c)
if [ -z "$ordered_wait_body" ] ||
  ! printf '%s\n' "$ordered_wait_body" | rg -Fq 'cuEventSynchronize'; then
  printf '%s\n' 'ordered completion wait implementation is missing' >&2
  failed=1
elif matches=$(printf '%s\n' "$ordered_wait_body" | rg -n \
  'moonbit_make|moonbit_incref|moonbit_decref|malloc|calloc|realloc|free' \
  2>/dev/null); then
  printf '%s\n%s\n' \
    'ordered completion wait must not allocate or retain caller state:' \
    "$matches" >&2
  failed=1
fi

peer_body=$(sed -n \
  '/^int32_t lf_cuda_device_can_access_peer_with_api(/,/^}/p' \
  internal/cuda/peer_access.c)
if [ -z "$peer_body" ]; then
  printf '%s\n' 'directed CUDA peer query implementation is missing' >&2
  failed=1
elif matches=$(printf '%s\n' "$peer_body" | rg -n \
  'cuCtx|cuMem|cuModule|cuStream|cuDeviceEnablePeerAccess|malloc|calloc|realloc|free' \
  2>/dev/null); then
  printf '%s\n%s\n' \
    'peer inventory query acquired CUDA authority or allocated:' \
    "$matches" >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(context, allocation, status\)\nextern "c" fn raw_allocation_lease_create' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' \
    'allocation lease creation must borrow its context and allocation' >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(lease\)\nextern "c" fn raw_allocation_lease_close' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' 'allocation lease close must borrow its native owner' >&2
  failed=1
fi

if ! rg -q \
  '"stub-cc-flags": "-std=c11 -Wall -Wextra -Werror"' \
  internal/cuda/moon.pkg; then
  printf '%s\n' 'internal/cuda must compile stubs with the strict C11 gate' >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(context, allocation, source\)\nextern "c" fn raw_context_copy_fixed_to_device' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' \
    'fixed host transfer must borrow context, allocation, and source' >&2
  failed=1
fi

if ! rg -q -U \
  '#borrow\(context, allocation, destination\)\nextern "c" fn raw_context_copy_device_to_fixed' \
  internal/cuda/ffi.mbt; then
  printf '%s\n' \
    'fixed host readback must borrow context, allocation, and destination' >&2
  failed=1
fi

fixed_transfer_body=$(sed -n \
  '/^int32_t lunaflux_cuda_context_copy_fixed_to_device(/,/^}/p' \
  internal/cuda/transfers.c)
fixed_transfer_helper=$(sed -n \
  '/^static int32_t lf_copy_to_device_range(/,/^}/p' \
  internal/cuda/transfers.c)
if [ -z "$fixed_transfer_body" ] || [ -z "$fixed_transfer_helper" ]; then
  printf '%s\n' 'fixed host transfer ABI implementation is missing' >&2
  failed=1
elif matches=$(printf '%s\n%s\n' \
  "$fixed_transfer_helper" "$fixed_transfer_body" | rg -n \
  'moonbit_make|moonbit_incref|moonbit_decref|malloc|calloc|realloc|free' \
  2>/dev/null); then
  printf '%s\n%s\n' \
    'fixed host transfer must not allocate or retain caller storage:' \
    "$matches" >&2
  failed=1
fi

fixed_readback_body=$(sed -n \
  '/^int32_t lunaflux_cuda_context_copy_device_to_fixed(/,/^}/p' \
  internal/cuda/transfers.c)
fixed_readback_helper=$(sed -n \
  '/^static int32_t lf_copy_from_device_range(/,/^}/p' \
  internal/cuda/transfers.c)
if [ -z "$fixed_readback_body" ] || [ -z "$fixed_readback_helper" ]; then
  printf '%s\n' 'fixed host readback ABI implementation is missing' >&2
  failed=1
elif matches=$(printf '%s\n%s\n' \
  "$fixed_readback_helper" "$fixed_readback_body" | rg -n \
  'moonbit_make|moonbit_incref|moonbit_decref|malloc|calloc|realloc|free' \
  2>/dev/null); then
  printf '%s\n%s\n' \
    'fixed host readback must not allocate or retain caller storage:' \
    "$matches" >&2
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux CUDA ABI ownership boundary is valid.'
