#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package='model/llama_tensor_parallel'
moon check "$package" --target native --release --deny-warn --warn-list +73
moon test "$package" --target native --release --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
if [ ! -f "$interface" ]; then
  printf '%s\n' 'Llama tensor-parallel generated interface is missing' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(scheduler|service|device|internal/cuda)|nccl|quant|compat' \
    "$package" --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'Llama rank planning imported runtime/backend authority' >&2
  exit 1
fi

if rg -n 'LlamaTensor(ParallelRankPlan|ParallelTopology|Placement|Extent|ArenaRegion|TransferSegment|TransferRecipe|RankPlacement|CollectiveSite)' \
    "$interface"; then
  printf '%s\n' 'Llama builder reintroduced family-owned physical types' >&2
  exit 1
fi

if ! rg -F -q \
    '@tensor_parallel_plan.TensorParallelRankPlan raise LlamaTensorParallelError' \
    "$interface"; then
  printf '%s\n' 'Llama builder does not return the generic rank plan' >&2
  exit 1
fi

for required in \
  '@tensor_parallel_plan.TensorParallelTopology::new(' \
  '@tensor_parallel_plan.TensorCollectiveSite::new(' \
  'placement != Replicated' \
  'local_bytes >= full_bytes' \
  'ShardColumns =>' \
  'segment_count=segments'; do
  if ! rg -F -q "$required" "$package" --glob '*.mbt'; then
    printf 'Llama tensor-parallel invariant is missing: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'FixedArray\[Byte\]|Bytes|Array\[Byte\]|copy_from|copy_to|allocate' \
    "$package" --glob '*.mbt' --glob '!**/*_test.mbt'; then
  printf '%s\n' 'Llama rank plan materializes or transfers tensor payload' >&2
  exit 1
fi

for file in $(rg --files "$package" -g '*.mbt'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'Llama tensor-parallel source exceeds file budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux Llama tensor-parallel boundary/source gate passed.'
