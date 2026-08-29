#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package='engine/tensor_parallel_device_plan'
moon check "$package" --target native --release --deny-warn --warn-list +73
moon test "$package" --target native --release --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
if [ ! -f "$interface" ]; then
  printf '%s\n' 'tensor-parallel device-plan interface evidence is missing' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(scheduler|service|internal|device$|engine/device_step|engine/rank_group)|nccl' \
    "$package" --glob '*.mbt' "$package/moon.pkg"; then
  printf '%s\n' 'tensor-parallel device plan imported runtime authority' >&2
  exit 1
fi

for required in \
  'pub fn build(' \
  'planned_rank_plan != materialized_rank_plan' \
  'model_plan.identity() != planned_rank_plan.identity()' \
  'topology.rank() != device.rank' \
  'topology.world_size() != device.world_size' \
  'device.target != expected_target' \
  'validate_region_order(planned_rank_plan)' \
  'validate_operation_layout(model_plan, planned_rank_plan)' \
  'geometry.kind() != operation.kind()' \
  'extent.row_count() >= extent.full_rows()' \
  'site.sequence().as_int() != index + 1' \
  'LastAxisAllGather'; do
  if ! rg -F -q "$required" "$package" --glob '*.mbt'; then
    printf 'tensor-parallel device-plan invariant is missing: %s\n' \
      "$required" >&2
    exit 1
  fi
done

for opaque in \
    TensorParallelRankDevice \
    TensorParallelMatrixView \
    TensorParallelDevicePlan; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${opaque} \\{\\n  // private fields\\n\\}" \
      "$interface"; then
    printf 'tensor-parallel device-plan type is not opaque: %s\n' \
      "$opaque" >&2
    exit 1
  fi
done

if rg -n '@device\.|Allocation|Context|Stream|Communicator|SchedulePlan|Request' \
    "$interface"; then
  printf '%s\n' 'tensor-parallel device-plan API leaked mutable authority' >&2
  exit 1
fi

for evidence in \
  'world one device plan is the full-shape oracle without collectives' \
  'multi rank views expose exact local projection and logit shapes' \
  'multi rank plan never references a complete sharded tensor region' \
  'generic tensor and operation views are exact ordered sources' \
  'collective sites preserve first and last semantic identities' \
  'device plan rejects model and materialized region identity drift' \
  'device plan rejects rank world and target substitution' \
  'public device plan binds authenticated file regions and topology'; do
  if ! rg -F -q "$evidence" "$package" --glob '*_wbtest.mbt'; then
    printf 'tensor-parallel device-plan evidence is missing: %s\n' \
      "$evidence" >&2
    exit 1
  fi
done

for accessor in \
    'pub fn TensorParallelDevicePlan::tensor_count' \
    'pub fn TensorParallelDevicePlan::tensor' \
    'pub fn TensorParallelDevicePlan::operation_count' \
    'pub fn TensorParallelDevicePlan::operation'; do
  if ! rg -F -q "$accessor" "$interface"; then
    printf 'tensor-parallel canonical tensor accessor is missing: %s\n' \
      "$accessor" >&2
    exit 1
  fi
done

for file in $(rg --files "$package" -g '*.mbt'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'tensor-parallel device-plan file exceeds budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux tensor-parallel device-plan gate passed.'
