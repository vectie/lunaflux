#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package='engine/tensor_parallel_worker_bootstrap'

moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon check "$package" --target native --release --deny-warn --warn-list +73
moon test "$package" --target native --release --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
if [ ! -f "$interface" ]; then
  printf '%s\n' 'tensor-parallel worker-bootstrap interface evidence is missing' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(scheduler|service|internal/(cuda|nccl)|runtime/descriptor)|LunaNexa|MoonGate' \
    "$package/moon.pkg"; then
  printf '%s\n' 'tensor-parallel worker bootstrap imported forbidden authority' >&2
  exit 1
fi

for required in \
  'kv_plan.rank_binding(rank)' \
  'kv_binding.digest().as_sha256()' \
  'contract.worker_limits() != reference.worker_limits()' \
  'contract.bootstrap_source_digest() != reference.bootstrap_source_digest()' \
  'contract.has_predecessor() != reference.has_predecessor()' \
  'contract.predecessor_value() != reference.predecessor_value()' \
  'max_query_tokens=startup.worker_limits().max_plan_tokens()' \
  'inspection.rank_plan()' \
  'authenticate_rank_plan' \
  'expected_device_plan != device_plan' \
  'plan.tensor_count()' \
  'plan.tensor(reference)' \
  'plan.operation_count()' \
  'plan.operation(operation)' \
  'lunaflux.tensor-parallel-device-plan.v2' \
  'execution_plans.length() != world_size' \
  'execution_plan.digest().as_sha256()' \
  'claim.execution_plan_digest_sha256' \
  'lunaflux.tensor-parallel-group-bootstrap.v4' \
  'admit_runtime_policy(source, topology)' \
  '@tensor_parallel_kv_plan.admit_from_topology(' \
  'source.execution()' \
  'policy.minimum_collective_runtime_version_code()' \
  'policy.maximum_collective_runtime_version_code()' \
  'claims.length() != world_size' \
  'seen[claim.rank]' \
  'rendezvous_bytes.length() != 128' \
  'const RANK_ENVELOPE_VERSION : UInt = 3' \
  'const RANK_ENVELOPE_LENGTH : Int = 752' \
  'frame_read_u32(bytes, RANK_ENVELOPE_CHECKSUM_OFFSET)' \
  'self.0.epoch != self.1'; do
  if ! rg -F -q "$required" "$package" --glob '*.mbt'; then
    printf 'tensor-parallel worker-bootstrap invariant is missing: %s\n' \
      "$required" >&2
    exit 1
  fi
done

for opaque in \
    TensorParallelGroupBootstrap \
    TensorParallelRankBootstrapEnvelope \
    TensorParallelGroupBootstrapDigest \
    TensorParallelTopologyDigest \
    TensorParallelWorkerContractDigest \
    TensorParallelDevicePlanDigest \
    TensorParallelKvBindingDigest; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${opaque} \\{\\n  // private fields\\n\\}" \
      "$interface"; then
    printf 'tensor-parallel worker-bootstrap type is not opaque: %s\n' \
      "$opaque" >&2
    exit 1
  fi
done

if rg -n 'ApprovedRoot|ApprovedRelativeLocator|Allocation|Context|Stream|Communicator|SchedulePlanBuffer|PlanBufferIdentity|RequestId|device name' \
    "$interface"; then
  printf '%s\n' 'tensor-parallel worker-bootstrap API leaked mutable or filesystem authority' >&2
  exit 1
fi

if rg -n 'TensorParallelAotLaunchContract|KernelArtifactBundle' "$interface"; then
  printf '%s\n' 'tensor-parallel worker-bootstrap API leaked launch or artifact authority' >&2
  exit 1
fi
if ! rg -q '@tensor_parallel_execution_plan.TensorParallelExecutionPlan' \
    "$interface" ||
  ! rg -q 'execution_plan_digest_sha256\(Self\) -> String' "$interface"; then
  printf '%s\n' 'tensor-parallel worker-bootstrap API lost execution-plan identity binding' >&2
  exit 1
fi

if sed -n '1,/^}/p' "$package/moon.pkg" | \
    rg -n 'llama_tensor_parallel|model/llama'; then
  printf '%s\n' 'tensor-parallel worker bootstrap imports a model-family plan' >&2
  exit 1
fi
if rg -n --pcre2 \
  'llama_tensor_parallel|LlamaTensor(?!ParallelPagedAotV5)|authenticate_llama' \
  "$package" \
  --glob '*.mbt' \
  --glob '!**/*_test.mbt' \
  --glob '!**/*_wbtest.mbt'; then
  printf '%s\n' 'tensor-parallel worker bootstrap retained model-family rank-plan vocabulary' >&2
  exit 1
fi

if rg -n 'append_layer|plan\.embedding\(\)|plan\.final_norm\(\)|plan\.logits\(\)|plan\.layer\(' \
    "$package/contract_digest.mbt"; then
  printf '%s\n' 'tensor-parallel worker bootstrap digest retained family-shaped device-plan access' >&2
  exit 1
fi

for evidence in \
  'group admission is rank-order canonical with rank-specific identities' \
  'group admission rejects duplicate incomplete and substituted rank sets' \
  'rank-local KV substitution changes the canonical group identity' \
  'execution-plan substitution changes group identity and malformed identity fails' \
  'device-plan digest binds canonical generic tensor order' \
  'public group admit authenticates execution plans and rejects worker or launch-artifact substitution' \
  'public admission rejects mixed worker sequence and query-limit domains' \
  'v5 source policy admits exact topology-bound runtime capacity' \
  'ordinary source cannot authorize tensor-parallel runtime' \
  'fixed rank envelope codec round trips every bounded identity' \
  'frame rejects corruption oversized lengths and stale epochs' \
  'frame destination bound fails before copying'; do
  if ! rg -F -q "$evidence" "$package" --glob '*_wbtest.mbt'; then
    printf 'tensor-parallel worker-bootstrap evidence is missing: %s\n' \
      "$evidence" >&2
    exit 1
  fi
done

for file in $(rg --files "$package" -g '*.mbt'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'tensor-parallel worker-bootstrap file exceeds budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux tensor-parallel worker-bootstrap gate passed.'
