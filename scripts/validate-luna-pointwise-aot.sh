#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package="$root/kernels/luna_cuda_pointwise_aot"

test -f "$package/moon.pkg"

if find "$package" -type f \( -name '*.mbt' -o -name '*.mbt.md' \) -exec wc -l {} + |
  awk '$2 != "total" && $1 > 500 { print; failed = 1 } END { exit failed }'
then
  :
else
  echo "pointwise AOT file exceeds 500-line debt alarm" >&2
  exit 1
fi

if rg -n 'internal/cuda|internal/process|runtime/approved_fs|device/ordered_executor|compiler/JIT|runtime JIT' "$package" --glob '*.mbt' --glob 'moon.pkg'
then
  echo "pointwise AOT package crossed an offline lowering boundary" >&2
  exit 1
fi

rg -q 'manifest_bindable=false' "$package/README.mbt.md"
rg -q 'manifest_bindable : Bool' "$package/types.mbt"
rg -q 'bind_manifest_pointwise_cuda_aot' "$package/bind_manifest.mbt"
rg -Fq 'contracts.scope() != FullGraph' "$package/bind_manifest.mbt"
rg -Fq 'contract.dimensions() != candidate.dimensions' "$package/bind_manifest.mbt"
rg -Fq 'contract.operands() != candidate.operands' "$package/bind_manifest.mbt"
rg -q 'reassociate=false' "$package/source_common.mbt"
rg -q 'sha256/' "$package/digest.mbt"

moon check kernels/luna_cuda_pointwise_aot --target native --deny-warn
moon test kernels/luna_cuda_pointwise_aot --target native --deny-warn

echo "Luna BF16 pointwise AOT boundary: PASS"
