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

for native_stub in cublas.c gemm.c loader.c modules.c resources.c transfers.c; do
  if ! rg -q "\"$native_stub\"" internal/cuda/moon.pkg; then
    printf '%s: %s\n' \
      'internal/cuda native-stub list is missing' "$native_stub" >&2
    failed=1
  fi
done

if ! rg -q \
  '"stub-cc-flags": "-std=c11 -Wall -Wextra -Werror"' \
  internal/cuda/moon.pkg; then
  printf '%s\n' 'internal/cuda must compile stubs with the strict C11 gate' >&2
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux CUDA ABI ownership boundary is valid.'
