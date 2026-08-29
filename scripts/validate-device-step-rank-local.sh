#!/bin/sh
set -eu

package_dir="engine/device_step"
source="$package_dir/prepare.mbt"

if ! rg -Fq 'pub fn prepare_rank_local(' "$source" ||
  ! rg -Fq '@tensor_parallel_kv_plan.TensorParallelKvPlan' "$source"; then
  echo "rank-local device-step constructor or KV authority input is missing" >&2
  exit 1
fi

for evidence in \
  'kv_plan.require_current_generation(model_generation)' \
  'kv_plan.require_current_contract(' \
  'kv_plan.rank_layout(rank)' \
  'kv_plan.global_kv_head_count() != global_heads' \
  'kv_plan.local_kv_head_count() != global_heads / world' \
  'layout.is_canonical_for(model_plan.identity(), local_geometry)'; do
  if ! rg -Fq "$evidence" "$source"; then
    printf 'rank-local device-step authentication is missing: %s\n' \
      "$evidence" >&2
    exit 1
  fi
done

common_body="$(sed -n '/^fn prepare_validated(/,/^}$/p' "$source")"
allocation_count="$(printf '%s\n' "$common_body" | rg -c 'allocate_one\(')"
if [ "$allocation_count" -ne 7 ]; then
  printf 'common device-step constructor must own exactly seven allocations, found %s\n' \
    "$allocation_count" >&2
  exit 1
fi

for stage in Counts TokenIds TokenPositions QueryRowOffsets SequenceLengths \
  PageOffsets PhysicalPageIndices; do
  stage_count="$(printf '%s\n' "$common_body" | rg -c \
    "(^|[^A-Za-z])${stage}([^A-Za-z]|$)")"
  if [ "$stage_count" -ne 1 ]; then
    printf 'common device-step constructor must allocate %s exactly once, found %s\n' \
      "$stage" "$stage_count" >&2
    exit 1
  fi
done

if [ "$(printf '%s\n' "$common_body" | rg -c 'host: make_host_buffers\(limits\)')" -ne 1 ]; then
  echo "common device-step constructor must create one seven-buffer host staging set" >&2
  exit 1
fi

outside_common="$(sed '/^fn prepare_validated(/,/^}$/d' "$source")"
if printf '%s\n' "$outside_common" | rg -n '^  let .* = allocate_one\('; then
  echo "device-step allocation escaped the common constructor" >&2
  exit 1
fi

delegate_count="$(rg -c '^  prepare_validated\($' "$source")"
if [ "$delegate_count" -ne 2 ]; then
  printf 'both public constructors must delegate once, found %s calls\n' \
    "$delegate_count" >&2
  exit 1
fi

cleanup_count="$(printf '%s\n' "$common_body" | rg -c \
  'construction_failed\(resources, error\)')"
if [ "$cleanup_count" -ne 7 ] ||
  printf '%s\n' "$outside_common" | rg -n 'return construction_failed\('; then
  echo "seven-buffer cleanup handling is duplicated or incomplete" >&2
  exit 1
fi

if ! rg -Fq 'test "cleanup failure retains retry authority and success is idempotent"' \
  "$package_dir/transaction_wbtest.mbt"; then
  echo "shared device-step cleanup retry parity test is missing" >&2
  exit 1
fi

test_source="$package_dir/rank_local_binding_wbtest.mbt"
for evidence in \
  'constructed_local_layout' \
  'validate_plan_binding(model_plan, local_layout, limits)' \
  'model_plan, full_layout, constructed_local_geometry, limits' \
  'ModelPlanGeneration::from_wire(8UL)' \
  'for rank in [-1, 2]' \
  'max_sequence_tokens=17'; do
  if ! rg -Fq "$evidence" "$test_source"; then
    printf 'rank-local hostile test evidence is missing: %s\n' \
      "$evidence" >&2
    exit 1
  fi
done

if rg -n 'vectie/lunaflux/(scheduler|internal/(cuda|nccl)|engine/(device_worker|tensor_parallel_collective|rank_group))' \
  "$package_dir/moon.pkg"; then
  echo "device-step rank-local binding acquired forbidden owner authority" >&2
  exit 1
fi

for file in "$package_dir"/*.mbt; do
  lines="$(wc -l < "$file" | tr -d ' ')"
  if [ "$lines" -ge 500 ]; then
    printf '%s exceeds the strict 499-line device-step budget (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

api="$package_dir/pkg.generated.mbti"
if [ -f "$api" ]; then
  if rg -n 'prepare_rank_local.*(@device_topology|DeviceWeights|Collective|Communicator|RankGroup)' \
    "$api"; then
    echo "rank-local device-step API leaks unrelated owner authority" >&2
    exit 1
  fi
fi

echo "device-step rank-local constructor boundaries: ok"
