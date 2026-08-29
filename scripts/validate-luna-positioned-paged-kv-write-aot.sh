#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

package=kernels/luna_cuda_paged_kv_write_aot

for source in "$package"/*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  [ "$lines" -lt 500 ] || {
    printf 'positioned paged KV-write source exceeds file budget: %s (%s)\n' \
      "$source" "$lines" >&2
    exit 1
  }
done

for required in \
  'pub fn lower_positioned_paged_kv_write_cuda_aot_candidate(' \
  'PositionedQkvPagedWriteBf16V1' \
  'const __nv_bfloat16* positioned_qkv' \
  '(long long)rows != (long long)prefill + (long long)decode' \
  'logical_page < 0 || logical_page >= end - begin' \
  'atomicAdd(dispatch_canary' \
  'fallback_family=monolithic-paged-write-and-attend' \
  'pub fn bind_positioned_paged_kv_write_compiled_artifact(' \
  'pub fn export_positioned_paged_kv_write_qualification(' \
  'pub fn evaluate_positioned_paged_kv_write_fake_campaign(' \
  'guard_silent_return_rejected=true' \
  'bounds_authority=runtime-validated-step-descriptor' \
  'manifest_bindable=false' \
  'promotion_authority=false'; do
  rg -F -q "$required" "$package" --glob '*.mbt' || {
    printf 'positioned paged KV-write boundary missing: %s\n' "$required" >&2
    exit 1
  }
done

if rg -n \
  'engine/|scheduler|service/|internal/cuda|extern[[:space:]]+"[cC]"|nvrtc|load_module|launch_synchronous' \
  "$package" --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
  printf '%s\n' 'positioned paged KV-write candidate gained runtime authority' >&2
  exit 1
fi

if rg -n \
  '^pub fn (PagedKvWriteCudaCandidate|PagedKvWriteCompiledArtifact|PagedKvWriteQualificationExport)::(new|from_|open|load|execute|promote)' \
  "$package" --glob '*.mbt' --glob 'pkg.generated.mbti' >/dev/null; then
  printf '%s\n' 'positioned paged KV-write authority became fabricable' >&2
  exit 1
fi

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
grep -F 'pub fn lower_positioned_paged_kv_write_cuda_aot_candidate(' \
  "$interface" >/dev/null
grep -F 'pub fn bind_positioned_paged_kv_write_compiled_artifact(' \
  "$interface" >/dev/null
grep -F 'pub fn export_positioned_paged_kv_write_qualification(' \
  "$interface" >/dev/null
grep -F 'pub fn evaluate_positioned_paged_kv_write_fake_campaign(' \
  "$interface" >/dev/null
for opaque in PagedKvWriteCudaCandidate PagedKvWriteCompiledArtifact; do
  rg -U -q "pub struct $opaque \\{\\n  // private fields\\n\\}" \
    "$interface"
done

printf '%s\n' 'positioned paged KV-write qualification-only AOT gate passed'
