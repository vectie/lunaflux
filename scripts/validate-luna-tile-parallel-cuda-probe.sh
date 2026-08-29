#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
package=tests/luna_tile_parallel_cuda_probe

for file in "$package"/*.mbt; do
  lines=$(wc -l <"$file")
  [ "$lines" -lt 500 ] || {
    echo "LunaTile physical probe file reached 500 lines: $file" >&2
    exit 1
  }
done

for anchor in \
  'specialize_luna_tile_parallel_candidate' \
  'scalar_oracle=exact-input-bytes' \
  'manifest_bindable=false' \
  'parallel_actual == expected && parallel_actual == serial_actual' \
  'compute_major() == 12 && capability.compute_minor() == 0' \
  'CUDA resource balance is nonzero'; do
  rg -Fq "$anchor" "$package" || {
    echo "LunaTile physical probe anchor missing: $anchor" >&2
    exit 1
  }
done

if rg -ni 'nvrtc|runtime.{0,16}compil|promotion_authority=(present|granted)|manifest_bindable=true' \
  "$package" >/dev/null; then
  echo 'LunaTile probe gained JIT, runtime compilation, manifest, or promotion authority' >&2
  exit 1
fi

moon fmt --check "$package"
moon check --target native --deny-warn --warn-list +73 "$package"
moon build --target native --deny-warn --warn-list +73 "$package"

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunatile-probe-validator.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT HUP INT TERM
probe=_build/native/debug/build/tests/luna_tile_parallel_cuda_probe/luna_tile_parallel_cuda_probe.exe
"$probe" export "$scratch" >"$scratch.out" 2>"$scratch.err"
[ ! -s "$scratch.err" ]
grep -Eq '^outcome=lunatile-parallel-sm120-fixture-exported .* authority=qualification-only$' \
  "$scratch.out"
[ "$(find "$scratch" -type f | wc -l | tr -d ' ')" -eq 5 ]
grep -F 'extern "C" __global__ void lunaflux_lunatile_parallel_simt_v1_ep_1202' \
  "$scratch/parallel.cu" >/dev/null
grep -F '__pipeline_memcpy_async' "$scratch/parallel.cu" >/dev/null
grep -F '__pipeline_wait_prior(0)' "$scratch/parallel.cu" >/dev/null
grep -Fx 'parallel_shared_memory_bytes=1024' "$scratch/fixture.v1" >/dev/null

echo 'LunaTile parallel sm120 exporter/public-device probe boundary passed.'
