#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 ABSOLUTE_NVCC_13_1 [CYCLES]" >&2
  exit 2
fi
nvcc=$1
cycles=${2:-8}
case "$nvcc" in /*) ;; *) echo 'nvcc path must be absolute' >&2; exit 2;; esac
[ -x "$nvcc" ] || { echo 'nvcc is not executable' >&2; exit 2; }
case "$cycles" in *[!0-9]*|'') echo 'cycles must be an integer' >&2; exit 2;; esac
[ "$cycles" -ge 1 ] && [ "$cycles" -le 32 ] || {
  echo 'cycles must be in 1..32' >&2
  exit 2
}
"$nvcc" --version | grep -F 'release 13.1,' >/dev/null || {
  echo 'probe requires nvcc release 13.1' >&2
  exit 2
}

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
scripts/validate-paged-bf16-shape-matrix-probe.sh >/dev/null
work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-paged-bf16-matrix.XXXXXX")
cleanup() { rm -rf -- "$work"; }
trap cleanup EXIT HUP INT TERM
mkdir -p "$work/sha256"
: >"$work/compile.stdout"
: >"$work/compile.stderr"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

compile_pair() {
  source=$1
  stem=$2
  first="$work/$stem.first.cubin"
  second="$work/$stem.second.cubin"
  for output in "$first" "$second"; do
    "$nvcc" -std=c++17 -O3 --fmad=false --prec-div=true --prec-sqrt=true \
      --ftz=false --maxrregcount=128 -arch=sm_120 --cubin \
      "$source" -o "$output" \
      >>"$work/compile.stdout" 2>>"$work/compile.stderr"
  done
  cmp -s "$first" "$second" || {
    echo "offline CUBIN builds differ: $source" >&2
    exit 1
  }
  magic=$(od -An -tx1 -N4 "$first" | tr -d ' \n')
  [ "$magic" = 7f454c46 ] || {
    echo "nvcc did not produce an ELF CUBIN: $source" >&2
    exit 1
  }
  digest=$(sha256_file "$first")
  target="$work/sha256/$digest.cubin"
  cp "$first" "$target"
  printf '%s\n' "$target"
}

pointwise=$(compile_pair \
  tests/paged_bf16_graph_probe/fixtures/graph_pointwise.cu pointwise)
qkv=$(compile_pair \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/qkv.cu qkv)
dense=$(compile_pair \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/dense.cu dense)
mlp=$(compile_pair \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/mlp.cu mlp)
lm_head=$(compile_pair \
  kernels/luna_cuda_projection_aot/fixtures/physical_sm120/lm_head.cu lm_head)
paged=$(compile_pair \
  kernels/luna_cuda_paged_attention_aot/fixtures/generated_reference_v1_ep3.cu paged)

[ ! -s "$work/compile.stderr" ] || {
  echo 'nvcc emitted unexpected stderr' >&2
  sed -n '1,120p' "$work/compile.stderr" >&2
  exit 1
}

if ! moon build --target native tests/paged_bf16_shape_matrix_probe --deny-warn \
  >"$work/build.stdout" 2>"$work/build.stderr"; then
  echo 'MoonBit shape-matrix probe build failed' >&2
  sed -n '1,120p' "$work/build.stdout" >&2
  sed -n '1,120p' "$work/build.stderr" >&2
  exit 1
fi
probe=$repo_root/_build/native/debug/build/tests/paged_bf16_shape_matrix_probe/paged_bf16_shape_matrix_probe.exe
[ -x "$probe" ] || {
  echo 'MoonBit shape-matrix executable is missing' >&2
  exit 1
}

"$probe" \
  "$repo_root/tests/paged_bf16_shape_matrix_probe/fixtures/graph_contract.v1" \
  "$pointwise" "$qkv" "$dense" "$mlp" "$lm_head" "$paged" "$cycles" \
  >"$work/runtime.stdout" 2>"$work/runtime.stderr"
[ ! -s "$work/runtime.stderr" ] || {
  echo 'shape-matrix probe emitted unexpected stderr' >&2
  sed -n '1,120p' "$work/runtime.stderr" >&2
  exit 1
}
launches=$((cycles * 5))
grep -Eq "^outcome=paged-bf16-shape-matrix-sm120-pass cycles=$cycles cases=5 graph_launches=$launches kernels=12 max_abs_error=[0-9.eE+-]+ capture=required artifacts=authenticated resources=closed scope=synthetic-shape-matrix$" \
  "$work/runtime.stdout"
cat "$work/runtime.stdout"
