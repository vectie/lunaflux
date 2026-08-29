#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 ABSOLUTE_NVCC_13_1" >&2
  exit 2
fi

nvcc=$1
case "$nvcc" in
  /*) ;;
  *) echo "nvcc path must be absolute" >&2; exit 2 ;;
esac
[ -x "$nvcc" ] || { echo "nvcc is not executable" >&2; exit 2; }
[ "$(readlink -f -- "$nvcc")" = "$nvcc" ] || {
  echo "nvcc path must be canonical" >&2
  exit 2
}
"$nvcc" --version | grep -F 'release 13.1,' >/dev/null || {
  echo "probe requires nvcc release 13.1" >&2
  exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
pointwise="$repo_root/kernels/luna_cuda_pointwise_aot/fixtures/physical_sm120"
projection="$repo_root/kernels/luna_cuda_projection_aot/fixtures/physical_sm120"

(cd "$pointwise" && sha256sum --check --strict SHA256SUMS)
(cd "$projection" && sha256sum --check --strict SHA256SUMS)

work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-bf16-family-probe.XXXXXX")
cleanup() {
  rm -rf -- "$work"
}
trap cleanup EXIT HUP INT TERM
compile_stderr="$work/compile.stderr"
: >"$compile_stderr"

compile_cubin() {
  source_file=$1
  output_file=$2
  "$nvcc" -std=c++17 -O3 --fmad=false --prec-div=true --prec-sqrt=true \
    --ftz=false --maxrregcount=128 -arch=sm_120 --cubin \
    "$source_file" -o "$output_file" 2>>"$compile_stderr"
}

compile_cubin "$pointwise/embedding.cu" "$work/embedding.cubin"
compile_cubin "$pointwise/rms_norm.cu" "$work/rms_norm.cubin"
compile_cubin "$pointwise/residual.cu" "$work/residual.cubin"
compile_cubin "$pointwise/rope.cu" "$work/rope.cubin"
compile_cubin "$projection/qkv.cu" "$work/qkv.cubin"
compile_cubin "$projection/dense.cu" "$work/dense.cubin"
compile_cubin "$projection/mlp.cu" "$work/mlp.cubin"
compile_cubin "$projection/lm_head.cu" "$work/lm_head.cubin"

"$nvcc" -std=c++17 -O2 -arch=sm_120 -Xcompiler=-Wall,-Wextra,-Werror \
  "$repo_root/scripts/luna-bf16-family-driver-probe.cpp" -lcuda \
  -o "$work/probe" 2>>"$compile_stderr"
[ ! -s "$compile_stderr" ] || {
  echo "nvcc emitted unexpected stderr" >&2
  sed -n '1,120p' "$compile_stderr" >&2
  exit 1
}

runtime_stdout="$work/runtime.stdout"
runtime_stderr="$work/runtime.stderr"
"$work/probe" \
  "$work/embedding.cubin" "$work/rms_norm.cubin" \
  "$work/residual.cubin" "$work/rope.cubin" \
  "$work/qkv.cubin" "$work/dense.cubin" \
  "$work/mlp.cubin" "$work/lm_head.cubin" \
  >"$runtime_stdout" 2>"$runtime_stderr"
[ ! -s "$runtime_stderr" ] || {
  echo "CUDA Driver probe emitted unexpected stderr" >&2
  sed -n '1,120p' "$runtime_stderr" >&2
  exit 1
}
grep -Fx 'outcome=bf16-family-sm120-pass families=8 live_tokens=3' \
  "$runtime_stdout" >/dev/null
cat "$runtime_stdout"
