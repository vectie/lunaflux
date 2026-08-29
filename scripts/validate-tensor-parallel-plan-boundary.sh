#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

generic='model/tensor_parallel_plan'
bridges=(
  'engine/tensor_parallel_device_plan'
  'engine/tensor_parallel_execution_plan'
  'internal/tensor_parallel_collective'
  'kernels/tensor_parallel_launch_contract'
)

if rg -n 'vectie/lunaflux/(model/llama_tensor_parallel|model/llama|device|engine|internal|kernels|scheduler|service)' \
    "$generic/moon.pkg"; then
  printf '%s\n' 'generic tensor-parallel plan imported family or runtime authority' >&2
  exit 1
fi

for package in "${bridges[@]}"; do
  if sed -n '1,/^}/p' "$package/moon.pkg" | \
      rg -n 'model/llama_tensor_parallel'; then
    printf 'production bridge imports Llama physical types: %s\n' "$package" >&2
    exit 1
  fi
  if rg -n -i 'llama_tensor_parallel|LlamaTensor|dense_llama|layer_count[[:space:]]*\*[[:space:]]*9' "$package" \
      --glob '*.mbt' \
      --glob '!**/*_test.mbt' \
      --glob '!**/*_wbtest.mbt'; then
    printf 'production bridge names Llama physical types: %s\n' "$package" >&2
    exit 1
  fi
  interface="$package/pkg.generated.mbti"
  if [ -f "$interface" ] && rg -n 'llama_tensor_parallel|LlamaTensor' "$interface"; then
    printf 'bridge interface exposes Llama physical types: %s\n' "$package" >&2
    exit 1
  fi
done

generic_interface="$generic/pkg.generated.mbti"
if [ -f "$generic_interface" ] && \
    rg -n 'llama_tensor_parallel|@device|@catalog|@engine|@internal|@scheduler|@service' \
      "$generic_interface"; then
  printf '%s\n' 'generic tensor-parallel interface exposes foreign authority' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux generic tensor-parallel plan boundary gate passed.'
