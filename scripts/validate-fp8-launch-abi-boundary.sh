#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

fail() {
  printf '%s\n' "FP8 launch ABI boundary: $1" >&2
  exit 1
}

package_dir=kernels/fp8_launch_abi
for file in "$package_dir"/moon.pkg "$package_dir"/*.mbt; do
  [ -f "$file" ] || fail "missing package source: $file"
  case "$file" in
    *.mbt)
      lines=$(wc -l < "$file" | tr -d ' ')
      [ "$lines" -lt 500 ] || fail "$file exceeds the 499-line budget"
      ;;
  esac
done

if rg -n 'internal/cuda|extern "c"|cuModule|cuLaunch|scheduler/|service/|runtime/' \
  "$package_dir" --glob '*.mbt' --glob moon.pkg; then
  fail 'host-only ABI crossed into CUDA, scheduler, service, or runtime authority'
fi
if rg -n '^pub fn [A-Za-z0-9_:]*(execute|load|compile|ready|dispatch)\(' \
  "$package_dir" --glob '*.mbt'; then
  fail 'public surface gained execution or readiness authority'
fi

for anchor in \
  'manifest.numeric_schema_digest() != plan.numeric_binding().digest()' \
  'dynamic_per_tensor_f32_v1()' \
  'profile~ : @launch_contract.KernelProfileId' \
  'validate_distinct_regions' \
  'RegionAlias' \
  'workspace_alignment' \
  'manifest.digest()' \
  'ReadOnlyArray::from_array(rows.to_owned())' \
  'AuthorityAcquired' \
  'AuthorityInUse' \
  'last_released_epoch' \
  'stateless multi-profile selection is exact' \
  'paged exact shape detaches caller row storage'
do
  rg -F -q "$anchor" "$package_dir" --glob '*.mbt' || \
    fail "missing invariant or hostile evidence: $anchor"
done

moon check kernels/fp8_launch_abi --target native --deny-warn --warn-list +73
moon test kernels/fp8_launch_abi --target native --deny-warn
printf '%s\n' 'FP8 dynamic-scale launch ABI boundary is valid.'
