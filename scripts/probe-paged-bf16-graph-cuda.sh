#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 ABSOLUTE_NVCC_13_1 [CYCLES]" >&2
  exit 2
fi
nvcc=$1
cycles=${2:-32}
case "$nvcc" in /*) ;; *) echo 'nvcc path must be absolute' >&2; exit 2;; esac
[ -x "$nvcc" ] || { echo 'nvcc is not executable' >&2; exit 2; }
case "$cycles" in *[!0-9]*|'') echo 'cycles must be an integer' >&2; exit 2;; esac
[ "$cycles" -ge 1 ] && [ "$cycles" -le 128 ] || {
  echo 'cycles must be in 1..128' >&2
  exit 2
}
"$nvcc" --version | grep -F 'release 13.1,' >/dev/null || {
  echo 'probe requires nvcc release 13.1' >&2
  exit 2
}

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
scripts/validate-paged-bf16-graph-probe.sh >/dev/null
work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-paged-bf16-graph.XXXXXX")
cleanup() { rm -rf -- "$work"; }
trap cleanup EXIT HUP INT TERM
compile_stderr=$work/compile.stderr
: >"$compile_stderr"

compile_cubin() {
  source=$1
  output=$2
  "$nvcc" -std=c++17 -O3 --fmad=false --prec-div=true --prec-sqrt=true \
    --ftz=false --maxrregcount=128 -arch=sm_120 --cubin \
    "$source" -o "$output" 2>>"$compile_stderr"
}

compile_cubin tests/paged_bf16_graph_probe/fixtures/graph_pointwise.cu "$work/pointwise.cubin"
compile_cubin kernels/luna_cuda_projection_aot/fixtures/physical_sm120/qkv.cu "$work/qkv.cubin"
compile_cubin kernels/luna_cuda_projection_aot/fixtures/physical_sm120/dense.cu "$work/dense.cubin"
compile_cubin kernels/luna_cuda_projection_aot/fixtures/physical_sm120/mlp.cu "$work/mlp.cubin"
compile_cubin kernels/luna_cuda_projection_aot/fixtures/physical_sm120/lm_head.cu "$work/lm_head.cubin"
compile_cubin kernels/luna_cuda_paged_attention_aot/fixtures/generated_reference_v1_ep3.cu "$work/paged.cubin"
[ ! -s "$compile_stderr" ] || {
  echo 'nvcc emitted unexpected stderr' >&2
  sed -n '1,120p' "$compile_stderr" >&2
  exit 1
}

for cubin in "$work/pointwise.cubin" "$work/qkv.cubin" "$work/dense.cubin" \
  "$work/mlp.cubin" "$work/lm_head.cubin" "$work/paged.cubin"; do
  magic=$(od -An -tx1 -N4 "$cubin" | tr -d ' \n')
  [ "$magic" = 7f454c46 ] || {
    echo "nvcc did not produce an ELF CUBIN: $cubin" >&2
    exit 1
  }
done

build_stdout=$work/build.stdout
build_stderr=$work/build.stderr
if ! moon build --target native tests/paged_bf16_graph_probe --deny-warn \
  >"$build_stdout" 2>"$build_stderr"; then
  echo 'MoonBit graph probe build failed' >&2
  sed -n '1,120p' "$build_stdout" >&2
  sed -n '1,120p' "$build_stderr" >&2
  exit 1
fi
probe=$repo_root/_build/native/debug/build/tests/paged_bf16_graph_probe/paged_bf16_graph_probe.exe
[ -x "$probe" ] || {
  echo 'MoonBit graph probe executable is missing' >&2
  exit 1
}

runtime_stdout=$work/runtime.stdout
runtime_stderr=$work/runtime.stderr
"$probe" \
  "$work/pointwise.cubin" "$work/qkv.cubin" "$work/dense.cubin" \
  "$work/mlp.cubin" "$work/lm_head.cubin" "$work/paged.cubin" "$cycles" \
  >"$runtime_stdout" 2>"$runtime_stderr"
[ ! -s "$runtime_stderr" ] || {
  echo 'complete graph probe emitted unexpected stderr' >&2
  sed -n '1,120p' "$runtime_stderr" >&2
  exit 1
}
grep -Eq "^outcome=paged-bf16-graph-sm120-pass cycles=$cycles kernels=12 mixed_rows=2 live_tokens=3 token=[0-9]+ max_abs_error=[0-9.eE+-]+ capture=required resources=closed$" \
  "$runtime_stdout"
cat "$runtime_stdout"
