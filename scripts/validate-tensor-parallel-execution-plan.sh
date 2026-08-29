#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package='engine/tensor_parallel_execution_plan'
moon check "$package" --target native --release --deny-warn --warn-list +73
moon test "$package" --target native --release --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
if [ ! -f "$interface" ]; then
  printf '%s\n' 'tensor-parallel execution-plan interface evidence is missing' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(scheduler|service|internal|engine/rank_group|engine/worker|device$)' \
    "$package/moon.pkg"; then
  printf '%s\n' 'tensor-parallel execution plan imported runtime authority' >&2
  exit 1
fi

if rg -n '@device\.|Allocation|Context|Stream|Communicator|Nccl|NCCL|cuda|cuBLAS|JIT|compiler|filesystem|ApprovedRoot' \
    "$interface"; then
  printf '%s\n' 'tensor-parallel execution-plan API leaked resource authority' >&2
  exit 1
fi

for required in \
  'validate_identity_surface(' \
  'artifacts.authenticates_tensor_parallel(' \
  'append_model_graph(encoder, model)' \
  'contract.operation_id().as_int() != operation_index' \
  'device_plan.tensor(reference)' \
  'view.local_rows()' \
  'view.local_columns()' \
  'assign_activation_slots(facts)' \
  'byte_count = byte_count.max(operand.byte_count())' \
  'InPlaceSumAllReduce' \
  'InPlaceLastAxisAllGather' \
  'pub fn TensorParallelExecutionPlan::operation_exact' \
  'pub fn TensorParallelExecutionPlan::collective_exact' \
  'Int64::from_int(launch_set.rank())' \
  'local_send_offset_bytes: send_offset' \
  'lunaflux.tensor-parallel-execution-plan.v1'; do
  if ! rg -F -q "$required" "$package" --glob '*.mbt'; then
    printf 'tensor-parallel execution-plan invariant is missing: %s\n' \
      "$required" >&2
    exit 1
  fi
done

release_c="_build/native/release/test/engine/tensor_parallel_execution_plan/tensor_parallel_execution_plan.whitebox_test.c"
if [ ! -f "$release_c" ]; then
  printf '%s\n' 'tensor-parallel execution-plan generated release C is missing' >&2
  exit 1
fi
for symbol in 'operation__exact' 'collective__exact'; do
  body=$(awk -v symbol="$symbol" '
    $0 ~ symbol "\\(" {
      signature = 1
      next
    }
    signature {
      if ($0 ~ /\) \{/) {
        signature = 0
        capture = 1
        next
      }
      if ($0 ~ /\);/) {
        signature = 0
      }
      next
    }
    capture {
      print
      if ($0 ~ /^  join_/) {
        exit
      }
    }
  ' "$release_c")
  if [ -z "$body" ]; then
    printf 'tensor-parallel execution-plan exact accessor symbol is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | rg -n 'moonbit_make_|moonbit_malloc|moonbit_gc_alloc'; then
    printf 'managed allocation found before exact-accessor rejection path: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

for evidence in \
  'execution plan authenticates exact physical spaces liveness artifacts and collectives' \
  'activation coloring never aliases overlapping liveness' \
  'digest_hex_difference(result.digest(), second_profile.digest()) >= 16' \
  'checked region arithmetic and limits reject hostile scalars'; do
  if ! rg -F -q "$evidence" "$package" --glob '*_wbtest.mbt'; then
    printf 'tensor-parallel execution-plan evidence is missing: %s\n' \
      "$evidence" >&2
    exit 1
  fi
done

for opaque in \
    TensorParallelExecutionPlan \
    TensorParallelExecutionPlanDigest; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${opaque} \\{\\n  // private fields\\n\\}" \
      "$interface"; then
    printf 'tensor-parallel execution-plan type is not opaque: %s\n' \
      "$opaque" >&2
    exit 1
  fi
done

for file in $(rg --files "$package" -g '*.mbt'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'tensor-parallel execution-plan file exceeds budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux tensor-parallel execution-plan gate passed.'
