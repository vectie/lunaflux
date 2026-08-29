#!/bin/sh
set -eu

package_dir="kernels/artifact"
api="$package_dir/pkg.generated.mbti"

if rg -n 'vectie/lunaflux/(engine|device$|internal/|scheduler|service|runtime|deploy)' \
  "$package_dir/admit.mbt" "$package_dir/types.mbt"; then
  echo "tensor-parallel artifact admission crossed an owner/backend boundary" >&2
  exit 1
fi

if ! rg -n 'fn required_tensor_parallel_entry_points\(' \
  "$package_dir/admit.mbt" >/dev/null ||
  ! rg -n 'pub fn admit_tensor_parallel\(' "$package_dir/admit.mbt" >/dev/null ||
  ! rg -n 'ensure_paged_catalog_version\(contracts.catalog_version\(\)\)' \
  "$package_dir/admit.mbt" >/dev/null; then
  echo "tensor-parallel artifact admission lost v3/requirement derivation" >&2
  exit 1
fi

tp_body=$(sed -n '/pub fn admit_tensor_parallel(/,/^}/p' "$package_dir/admit.mbt")
if printf '%s\n' "$tp_body" | rg -n 'Device(Context|Allocation|Stream)|Communicator|cu[A-Z]|cuda[A-Z]|JIT|Jit|Compiler|Root|Path'; then
  echo "tensor-parallel artifact admission gained resource/path/JIT authority" >&2
  exit 1
fi
if [ "$(printf '%s\n' "$tp_body" | rg -c 'admit_required\(')" -ne 1 ]; then
  echo "tensor-parallel artifact admission does not delegate exactly once" >&2
  exit 1
fi
if printf '%s\n' "$tp_body" | rg -n 'exact_digest|validate_module_inputs|validate_entry_inputs|own_module_input'; then
  echo "tensor-parallel artifact admission copied lower admission logic" >&2
  exit 1
fi

if [ ! -f "$api" ] ||
  ! rg -n 'pub fn admit_tensor_parallel\(' "$api" >/dev/null; then
  echo "tensor-parallel artifact generated API is missing" >&2
  exit 1
fi
if [ "$(rg -c '^pub struct KernelArtifactBundle \{' "$api")" -ne 1 ] ||
  rg -n 'TensorParallelArtifact(Bundle|Module|EntryPoint)' "$api"; then
  echo "tensor-parallel artifact admission introduced a second bundle surface" >&2
  exit 1
fi

for file in "$package_dir"/*.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "$file exceeds the strict 499-line artifact package budget" >&2
    exit 1
  fi
done

echo "tensor-parallel artifact boundaries: ok"
