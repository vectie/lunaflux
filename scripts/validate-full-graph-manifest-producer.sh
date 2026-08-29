#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package_dir="$repo_root/kernels/full_graph_manifest"

if rg -n \
  'internal/cuda|moonbitlang/async|moonbitlang/x/sys|luna_cuda_aot|extern[[:space:]]+"c"|@(fs|sys)\.' \
  "$package_dir" \
  --glob '*.mbt' \
  --glob 'moon.pkg'; then
  echo "full-graph manifest producer crossed its inert offline boundary" >&2
  exit 1
fi

find "$package_dir" -type f -name '*.mbt' -print | while IFS= read -r source; do
  lines=$(wc -l < "$source")
  if [ "$lines" -ge 500 ]; then
    echo "full-graph manifest producer source exceeds 499 lines: $source ($lines)" >&2
    exit 1
  fi
done

cd "$repo_root"
moon check kernels/full_graph_manifest --target native --deny-warn
moon test kernels/full_graph_manifest --target native --deny-warn

echo "full-graph manifest producer gate passed"
