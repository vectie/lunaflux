#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package='model/device_materialize'
moon check "$package" --target native --release --deny-warn --warn-list +73
moon test "$package" --target native --release --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
generated_c='_build/native/release/test/model/device_materialize/device_materialize.whitebox_test.c'
if [ ! -f "$interface" ] || [ ! -f "$generated_c" ]; then
  printf '%s\n' 'sharded device-weight release evidence is missing' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(scheduler|service|internal/cuda|engine/device_topology)|nccl' \
    "$package"/sharded*.mbt "$package/moon.pkg"; then
  printf '%s\n' 'sharded weight loading imported runtime/rank-group authority' >&2
  exit 1
fi

for required in \
  'pub fn inspect_sharded_file(' \
  'pub fn load_inspected_sharded_file(' \
  'rank~ : Int' \
  'world_size~ : Int' \
  'alignment_bytes~ : Int64' \
  'max_arena_bytes~ : Int64' \
  'let rank_plan = build_authenticated_sharded_rank_plan(' \
  'semantic_plan.identity().content() != model.content_digest()' \
  'assert_eq(prepared.rank_plan, wb_sharded_rank_plan(file, rank))' \
  'let source_order = sharded_source_order(rank_plan)' \
  'validate_sharded_transfers(' \
  'let digest = hash_stream(' \
  'let resource = open(prepared.rank_plan.arena_bytes())' \
  'transfer_sharded_streaming_load(' \
  'digest.update_from_iter(prepared.scratch[:count].iter())' \
  'finish_sharded_weight_file_load(' \
  'SourceClose'; do
  if ! rg -F -q "$required" "$package" --glob 'sharded*.mbt'; then
    printf 'sharded device-weight invariant is missing: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n --pcre2 -U \
    'pub fn inspect_sharded_file\((?:(?!\n\) ->)[\s\S])*TensorParallelRankPlan' \
    "$package" --glob 'sharded*.mbt'; then
  printf '%s\n' 'inspection accepts a source-bearing prebuilt rank plan' >&2
  exit 1
fi

if rg -n '\.segment\(' \
    "$package/sharded_transfer.mbt"; then
  printf '%s\n' 'sharded transfer cursor allocates per-segment record evidence' >&2
  exit 1
fi

if rg -n 'Bytes::|Array\[Byte\]::|FixedArray::make' \
    "$package/sharded_transfer.mbt"; then
  printf '%s\n' 'sharded transfer pass constructs payload or segment storage' >&2
  exit 1
fi

if ! rg -q 'source_order : FixedArray\[Int\]' \
    "$package/sharded_types.mbt" ||
  rg -q 'FixedArray\[AbsoluteShardedTransfer\]|Array\[AbsoluteShardedTransfer\]' \
    "$package" --glob 'sharded*.mbt'; then
  printf '%s\n' 'sharded source ordering is not O(tensor-count) scalar storage' >&2
  exit 1
fi

for opaque in \
    DeviceShardedWeightFileInspection \
    DeviceShardedWeights \
    FailedDeviceShardedWeightPreparation; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${opaque} \\{\\n  // private fields\\n\\}" \
      "$interface"; then
    printf 'sharded device-weight authority is not opaque: %s\n' "$opaque" >&2
    exit 1
  fi
done

for evidence in \
  'public sharded inspection and materialization preserve dense and paged plan identity' \
  'sharded inspection rejects foreign content and incompatible same-content graph' \
  'sharded stream copies row column and replicated ranges directly' \
  'all rank shard arenas reconstruct complete tensors' \
  'physically reordered payload tensors stream and reconstruct by source' \
  'sharded mutation and read failures close local allocation' \
  'sharded copy failure retains cleanup authority when close fails' \
  'sharded preflight authenticates complete file before allocation' \
  'sharded source close failure preserves retryable allocation ownership'; do
  if ! rg -F -q "$evidence" "$package" --glob '*_wbtest.mbt'; then
    printf 'sharded device-weight evidence is missing: %s\n' "$evidence" >&2
    exit 1
  fi
done

for file in $(rg --files "$package" -g 'sharded*.mbt'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'sharded device-weight source exceeds file budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux sharded device-weight materialization gate passed.'
