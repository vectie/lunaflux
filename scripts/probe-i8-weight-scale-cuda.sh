#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 ABSOLUTE_NVCC_13_1 sm_89|sm_90|sm_120 [CYCLES]" >&2
  exit 2
fi
nvcc=$1
target=$2
cycles=${3:-32}
case "$nvcc" in /*) ;; *) echo 'nvcc path must be absolute' >&2; exit 2;; esac
[ -x "$nvcc" ] || { echo 'nvcc is not executable' >&2; exit 2; }
case "$target" in
  sm_89|sm_90|sm_120) ;;
  *) echo 'target must be exact sm_89, sm_90, or sm_120' >&2; exit 2 ;;
esac
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
scripts/validate-i8-weight-scale-physical-probe.sh >/dev/null
work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-i8-weight-scale.XXXXXX")
cleanup() { rm -rf -- "$work"; }
trap cleanup EXIT HUP INT TERM

compile_stdout=$work/compile.stdout
compile_stderr=$work/compile.stderr
"$nvcc" -std=c++17 -O3 --fmad=false --prec-div=true --prec-sqrt=true \
  --ftz=false --maxrregcount=128 -arch="$target" --cubin \
  tests/i8_weight_scale_cuda_probe/fixtures/i8_weight_scale_projection.cu \
  -o "$work/i8_weight_scale.cubin" >"$compile_stdout" 2>"$compile_stderr"
[ ! -s "$compile_stdout" ] && [ ! -s "$compile_stderr" ] || {
  echo 'nvcc emitted unexpected output' >&2
  sed -n '1,120p' "$compile_stdout" >&2
  sed -n '1,120p' "$compile_stderr" >&2
  exit 1
}
magic=$(od -An -tx1 -N4 "$work/i8_weight_scale.cubin" | tr -d ' \n')
[ "$magic" = 7f454c46 ] || {
  echo 'nvcc did not produce an ELF CUBIN' >&2
  exit 1
}

build_stdout=$work/build.stdout
build_stderr=$work/build.stderr
if ! moon build --target native tests/i8_weight_scale_cuda_probe --deny-warn \
  >"$build_stdout" 2>"$build_stderr"; then
  echo 'MoonBit I8 probe build failed' >&2
  sed -n '1,120p' "$build_stdout" >&2
  sed -n '1,120p' "$build_stderr" >&2
  exit 1
fi
probe=$repo_root/_build/native/debug/build/tests/i8_weight_scale_cuda_probe/i8_weight_scale_cuda_probe.exe
[ -x "$probe" ] || {
  echo 'MoonBit I8 probe executable is missing' >&2
  exit 1
}

runtime_stdout=$work/runtime.stdout
runtime_stderr=$work/runtime.stderr
"$probe" "$work/i8_weight_scale.cubin" "$target" "$cycles" \
  >"$runtime_stdout" 2>"$runtime_stderr"
[ ! -s "$runtime_stderr" ] || {
  echo 'I8 physical probe emitted unexpected stderr' >&2
  sed -n '1,120p' "$runtime_stderr" >&2
  exit 1
}
target_digits=${target#sm_}
grep -Eq "^outcome=i8-weight-scale-physical-pass target=sm_${target_digits} cycles=${cycles} rows=3 inner=4 outputs=4 abi=bf16-i8-f32scale-bf16 catalog=v4 capture=required max_abs_error=0(\\.0+)? resources=closed scope=single-output-projection$" \
  "$runtime_stdout"
cat "$runtime_stdout"
