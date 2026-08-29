#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
package_dir="$repo_root/kernels/luna_cuda_projection_aot"
fixture_dir="$package_dir/fixtures/physical_sm120"

(cd "$fixture_dir" && shasum -a 256 -c SHA256SUMS)
if grep -En '^(module_sha256|family_id|workspace)=' "$fixture_dir"/*.recipe; then
  echo "projection candidate recipe contains final-contract identity" >&2
  exit 1
fi
for recipe in "$fixture_dir"/*.recipe; do
  grep -qx 'manifest_bindable=false' "$recipe"
done

if grep -Eq 'internal/cuda|/device"|core/(fs|process|stdio)|moonbitlang/x/(fs|process)' \
  "$package_dir/moon.pkg"; then
  echo "projection AOT package crosses the offline lowering boundary" >&2
  exit 1
fi

if grep -ERn 'nvrtc|cuModuleLoadDataEx|popen[[:space:]]*\(|system[[:space:]]*\(' \
  "$package_dir" --include='*.mbt' --include='moon.pkg'; then
  echo "projection AOT package contains a runtime compiler/process channel" >&2
  exit 1
fi

for source in "$package_dir"/*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  if [ "$lines" -gt 500 ]; then
    echo "projection AOT source exceeds 500 lines: $source ($lines)" >&2
    exit 1
  fi
done

cd "$repo_root"
moon check --target native --deny-warn kernels/luna_cuda_projection_aot
moon test --target native --deny-warn kernels/luna_cuda_projection_aot
echo "projection AOT boundary: pass"
