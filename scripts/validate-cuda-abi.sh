#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

if matches=$(rg -n \
  --glob 'moon.pkg' \
  --glob '!device/moon.pkg' \
  --glob '!internal/cuda/moon.pkg' \
  'vectie/lunaflux/internal/cuda' 2>/dev/null); then
  printf '%s\n%s\n' \
    'only device may import the private CUDA package:' \
    "$matches" >&2
  failed=1
fi

outside_cuda_files=$(rg --files \
  --glob '*.mbt' --glob '*.c' --glob '*.h' |
  sed '/^internal\/cuda\//d')
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
  'TODO|HACK|FIXME' 2>/dev/null); then
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

for native_stub in allocation_lease.c allocation_lease_probe.c cublas.c gemm.c launch.c loader.c modules.c region_probe.c regions.c resources.c transfer_probe.c transfers.c; do
  if ! rg -q "\"$native_stub\"" internal/cuda/moon.pkg; then
    printf '%s: %s\n' \
      'internal/cuda native-stub list is missing' "$native_stub" >&2
    failed=1
  fi
done

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
