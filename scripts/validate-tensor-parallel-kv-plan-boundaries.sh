#!/bin/sh
set -eu

package_dir="engine/tensor_parallel_kv_plan"

if rg -n 'BlockTable|PageAllocator([^L]|$)|allocate_one|allocate_run|release_|retain_|scheduler|internal/cuda|DeviceContext|DeviceAllocation|Stream|Communicator' "$package_dir" --glob '*.mbt' --glob '!**/*_wbtest.mbt'; then
  echo "tensor-parallel KV plan leaks owner/backend/scheduler authority" >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(kv/block_table|scheduler|internal/cuda|device$)' "$package_dir/moon.pkg"; then
  echo "tensor-parallel KV plan imports a forbidden mutable/backend package" >&2
  exit 1
fi

if ! rg -n 'ModelPlanGeneration' "$package_dir/admit.mbt" >/dev/null; then
  echo "tensor-parallel KV admission does not bind the canonical model generation" >&2
  exit 1
fi

if ! rg -n 'TensorParallelRuntimeAdmission' "$package_dir/admit.mbt" >/dev/null; then
  echo "tensor-parallel KV admission bypasses runtime capacity evidence" >&2
  exit 1
fi

if rg -n 'TensorParallelKvRankBinding::new|max_arena_bytes~' "$package_dir" --glob '*.mbt'; then
  echo "tensor-parallel KV plan defines a parallel caller capacity claim" >&2
  exit 1
fi

api="$package_dir/pkg.generated.mbti"
if [ ! -f "$api" ]; then
  echo "tensor-parallel KV generated API evidence is missing" >&2
  exit 1
fi
if rg -n 'TensorParallelKv(RankBinding|ContractDigest)::new|PageAllocator|BlockTableArena|DeviceContext|DeviceAllocation|Communicator' "$api"; then
  echo "tensor-parallel KV generated API leaks fabrication or owner authority" >&2
  exit 1
fi
if ! rg -n 'TensorParallelRuntimeAdmission.*TensorParallelDevicePlan' "$api" >/dev/null ||
  ! rg -n 'rank_binding\(Self, Int\).*TensorParallelKvRankBinding' "$api" >/dev/null ||
  ! rg -n 'rank_layout\(Self, Int\).*DeviceKvLayout' "$api" >/dev/null; then
  echo "tensor-parallel KV generated API lost authoritative admission or proof output" >&2
  exit 1
fi

if ! rg -n 'page\.index\(\)|page : @page_allocator\.PageId' "$package_dir/views.mbt" >/dev/null; then
  echo "logical PageId is not projected directly into rank-local offsets" >&2
  exit 1
fi

if ! rg -n 'FullArenaReplication' "$package_dir/admit.mbt" >/dev/null; then
  echo "multi-rank full KV replication is not rejected" >&2
  exit 1
fi

for file in "$package_dir"/*.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "$file exceeds the strict 499-line package budget" >&2
    exit 1
  fi
done

echo "tensor-parallel KV plan boundaries: ok"
