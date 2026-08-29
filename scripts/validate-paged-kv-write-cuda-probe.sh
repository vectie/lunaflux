#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
package=tests/paged_kv_write_cuda_probe
fail() { printf 'paged KV-write CUDA probe boundary failed: %s\n' "$1" >&2; exit 1; }

for dependency in \
  'vectie/lunaflux/device' \
  'vectie/lunaflux/kernels/luna_cuda_paged_kv_write_aot' \
  'vectie/lunaflux/release/luna_paged_kv_write_physical_evidence'; do
  grep -Fq "\"$dependency\"" "$package/moon.pkg" || fail "probe dependency missing: $dependency"
done
for anchor in \
  'PositionedQkvActivation::new' \
  'positioned_paged_kv_write_compile_receipt' \
  'builder receipt identity drifted' \
  'serial_cuda_oracle=pass' \
  'device_uuid=\{observation.device_uuid}' \
  'device_pci=\{observation.device_pci}' \
  '(long long)rows!=(long long)prefill+(long long)decode' \
  'logical_page>=end-begin' \
  'candidate dispatch canary mismatch' \
  'try candidate_function.close() catch' \
  'try candidate_module.close() catch' \
  'try arena.close() catch' \
  'resources=\{balance.render()}' \
  'authority=qualification-only'; do
  rg -Fq "$anchor" "$package" || fail "probe anchor missing: $anchor"
done
if rg -ni 'nvrtc|--ptx|\.ptx|runtime.{0,16}compil|manifest_bindable\([^)]*true|promotion_authority=(present|granted)|runtime.serv' "$package" --glob '*.mbt' >/dev/null; then
  fail 'probe gained JIT, runtime, manifest, or promotion authority'
fi
max_lines=$(find "$package" -type f -name '*.mbt' -exec wc -l {} + | awk '$2 != "total" {print $1}' | sort -nr | sed -n '1p')
[ "$max_lines" -lt 500 ] || fail 'probe source reached 500 lines'

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null
scripts/validate-luna-positioned-paged-kv-write-aot.sh >/dev/null
scripts/validate-cuda-abi.sh >/dev/null

printf '%s\n' 'Paged KV-write CUDA probe preserves exact positioned-QKV, oracle, cleanup, and qualification boundaries.'
