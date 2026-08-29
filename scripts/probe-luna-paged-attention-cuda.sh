#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 ABSOLUTE_NVCC SM_ARCH" >&2
  exit 2
fi

nvcc_path=$1
sm_arch=$2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
source_path="$script_dir/../kernels/luna_cuda_paged_attention_aot/fixtures/generated_reference_v1_ep3.cu"
function_symbol=lunaflux_paged_attention_bf16_reference_v1_ep_3

case "$nvcc_path" in
  /*) ;;
  *) echo "nvcc path must be absolute" >&2; exit 2 ;;
esac
case "$sm_arch" in
  sm_[0-9][0-9]|sm_[0-9][0-9][0-9]) ;;
  *) echo "SM_ARCH must look like sm_80 or sm_120" >&2; exit 2 ;;
esac

test -x "$nvcc_path"
test -f "$source_path"
test ! -L "$source_path"

probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-paged-attention-probe.XXXXXX")
trap 'rm -rf -- "$probe_dir"' EXIT HUP INT TERM

"$nvcc_path" -std=c++17 -cubin -arch="$sm_arch" -O3 --fmad=false \
  --maxrregcount=128 \
  "$source_path" -o "$probe_dir/paged_attention.cubin"
"$nvcc_path" -std=c++17 -O2 \
  "$script_dir/../kernels/luna_cuda_paged_attention_aot/fixtures/physical_probe.cu" \
  -lcuda -o "$probe_dir/physical_probe"
"$probe_dir/physical_probe" "$probe_dir/paged_attention.cubin" "$function_symbol"
