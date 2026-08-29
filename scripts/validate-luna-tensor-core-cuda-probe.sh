#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"

moon check tests/luna_tensor_core_cuda_probe --target native --deny-warn --warn-list +73
moon test tests/luna_tensor_core_cuda_probe --target native --deny-warn --warn-list +73
moon info tests/luna_tensor_core_cuda_probe --target native >/dev/null
moon build tests/luna_tensor_core_cuda_probe --target native --deny-warn --warn-list +73

probe=$root/_build/native/debug/build/tests/luna_tensor_core_cuda_probe/luna_tensor_core_cuda_probe.exe
scratch=$(mktemp -d /private/tmp/lunaflux-tensor-core-export.XXXXXX)
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM
mkdir "$scratch/first" "$scratch/second"
"$probe" export "$scratch/first" >"$scratch/first.stdout" 2>"$scratch/first.stderr"
"$probe" export "$scratch/second" >"$scratch/second.stdout" 2>"$scratch/second.stderr"
[ ! -s "$scratch/first.stderr" ] && [ ! -s "$scratch/second.stderr" ] || {
  printf '%s\n' 'tensor-core deterministic exporter emitted stderr' >&2
  exit 1
}
for file in fixture.v1 serial.cu serial.canonical tensor_core.cu tensor_core.canonical; do
  cmp -s "$scratch/first/$file" "$scratch/second/$file" || {
    printf 'tensor-core deterministic export drifted: %s\n' "$file" >&2
    exit 1
  }
done
if "$probe" export "$scratch/first" >"$scratch/overwrite.stdout" 2>"$scratch/overwrite.stderr"; then
  printf '%s\n' 'tensor-core exporter overwrote an existing fixture' >&2
  exit 1
fi
cleanup
trap - EXIT HUP INT TERM

for required in \
  'lunaflux_lunatile_wmma_bf16_m16n16k16_sm120_v1' \
  'cpu_oracle=independent-ordered-f32-v1' \
  'serial_cuda_oracle=required' \
  'absolute_tolerance=0.001' \
  'relative_tolerance=0.0001' \
  'maximum_registers_per_thread=128' \
  'manifest_bindable=false' \
  'promotion_authority=absent'; do
  rg -F -q "$required" tests/luna_tensor_core_cuda_probe --glob '*.mbt' || {
    printf 'tensor-core CUDA probe boundary missing: %s\n' "$required" >&2
    exit 1
  }
done

if rg -n 'RequireExternallyQualifiedTensorCore|manifest_bindable=true|promotion_authority=(present|granted)|NVRTC|nvrtc|runtime.?jit' \
  tests/luna_tensor_core_cuda_probe --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'tensor-core CUDA probe acquired runtime/promotion authority' >&2
  exit 1
fi

for file in $(rg --files tests/luna_tensor_core_cuda_probe -g '*.mbt'); do
  lines=$(wc -l <"$file")
  [ "$lines" -lt 500 ] || {
    printf 'tensor-core CUDA probe source exceeds file budget: %s (%s)\n' "$file" "$lines" >&2
    exit 1
  }
done

printf '%s\n' 'Luna tensor-core CUDA probe boundary passed'
