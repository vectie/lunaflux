#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

while read -r expected source; do
  [ -n "$source" ] || continue
  actual=$(sha256_file "$source")
  [ "$actual" = "$expected" ] || {
    echo "graph probe source digest mismatch: $source" >&2
    exit 1
  }
done < tests/paged_bf16_graph_probe/SOURCE_SHA256SUMS

if rg -n 'vectie/lunaflux/internal/cuda' \
  tests/paged_bf16_graph_probe/moon.pkg >/dev/null; then
  echo 'graph probe bypasses the public device boundary' >&2
  exit 1
fi

for anchor in \
  'OrderedKernelCaptureRequired' \
  'OrderedKernelCaptured' \
  'executor.launch_captured()' \
  'executor.record_completion()' \
  'executor.wait_completion()' \
  'executor.reset()'; do
  rg -Fq "$anchor" tests/paged_bf16_graph_probe/graph.mbt || {
    echo "graph lifecycle anchor is missing: $anchor" >&2
    exit 1
  }
done

[ "$(rg -o 'lunaflux_probe_(embedding|rms_norm|rope_qkv|residual)_h4' \
  tests/paged_bf16_graph_probe/fixtures/graph_pointwise.cu | sort -u | wc -l | tr -d ' ')" -eq 4 ] || {
  echo 'graph-compatible pointwise fixture is incomplete' >&2
  exit 1
}

for anchor in \
  'attention_bf16(rope, keys, values)' \
  'project_bf16(final_norm, fixture_bf16(20, 64), 3, 4, 16)' \
  'selected == expected_selected' \
  'compare_output(' \
  'embedding, norm1, qkv, rope, attention, keys, values, projected'; do
  rg -Fq "$anchor" tests/paged_bf16_graph_probe/cpu_referee.mbt || {
    echo "independent referee anchor is missing: $anchor" >&2
    exit 1
  }
done

if rg -n 'internal/cuda|extern\s+"[cC]"|#external' \
  --glob '*.mbt' tests/paged_bf16_graph_probe >/dev/null; then
  echo 'CPU referee depends on the CUDA implementation under test' >&2
  exit 1
fi

if rg -n 'nvrtc|\.ptx|--ptx|compute_[0-9]+,code=compute' \
  tests/paged_bf16_graph_probe scripts/probe-paged-bf16-graph-cuda.sh \
  >/dev/null 2>&1; then
  echo 'graph probe introduced a PTX or runtime compilation path' >&2
  exit 1
fi

echo 'LunaFlux synthetic paged-BF16 graph probe boundary passed.'
