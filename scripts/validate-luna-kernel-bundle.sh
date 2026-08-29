#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package_dir="$repo_root/kernels/luna_kernel_bundle"

if rg -n \
  'internal/cuda|moonbitlang/async|moonbitlang/x/sys|extern[[:space:]]+"[cC]"|@(fs|sys)\.' \
  "$package_dir" \
  --glob '*.mbt' \
  --glob 'moon.pkg'; then
  echo "luna kernel bundle crossed its inert offline boundary" >&2
  exit 1
fi

if rg -n \
  'nvcc|ptxas|cuModuleLoadData|runtime[_ -]jit|source[_ -]compiler' \
  "$package_dir" \
  --glob '*.mbt' \
  --glob 'moon.pkg'; then
  echo "luna kernel bundle contains a production compile or load channel" >&2
  exit 1
fi

find "$package_dir" -type f -name '*.mbt' -print | while IFS= read -r source; do
  lines=$(wc -l < "$source")
  if [ "$lines" -ge 500 ]; then
    echo "luna kernel bundle source exceeds 499 lines: $source ($lines)" >&2
    exit 1
  fi
done

cd "$repo_root"
moon check kernels/luna_kernel_bundle --target native --deny-warn
moon test kernels/luna_kernel_bundle --target native --deny-warn

echo "luna kernel bundle gate passed"
