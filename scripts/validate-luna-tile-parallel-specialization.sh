#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

moon check --target native --deny-warn --warn-list +73 -p kernels/luna_tile_ir
moon test --target native --deny-warn --warn-list +73 -p kernels/luna_tile_ir

if rg -n '@(fs|process|sys|async)|dlopen|cuModule|cudaLaunch|manifest\.write' \
  kernels/luna_tile_ir/parallel_*.mbt
then
  echo "parallel LunaTile specialization crossed its inert authority boundary" >&2
  exit 1
fi

for source in kernels/luna_tile_ir/parallel_*.mbt
do
  lines=$(wc -l < "$source")
  if [ "$lines" -ge 500 ]
  then
    echo "$source exceeds the focused-file limit: $lines" >&2
    exit 1
  fi
done

rg -q 'tile_mapping=round_major:block,warp,lane' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -q 'serial_oracle_role=qualification_only' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -q 'manifest_bindable=false' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -q '__pipeline_wait_prior' kernels/luna_tile_ir/parallel_source.mbt
rg -q 'global_read_regions\[read_index\] != global_write_regions\[write_index\]' \
  kernels/luna_tile_ir/parallel_validate.mbt
rg -q 'parallel specialization rejects cross-worker global read-write alias' \
  kernels/luna_tile_ir/parallel_cross_write_test.mbt

echo "LunaTile inert parallel-specialization boundary: PASS"
