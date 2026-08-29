#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

admission=runtime/tensor_parallel_admission
report=ops/tensor_parallel_report

if rg -n \
  'vectie/lunaflux/(runtime/(descriptor_file|instance_admission)|config/(device|runtime_resolved)|engine/(worker|device_plan|device_step)|scheduler|service|model|internal/(cuda|nccl))' \
  "$admission/moon.pkg" "$report/moon.pkg"; then
  printf '%s\n' \
    'tensor-parallel capacity admission widened an existing runtime owner' >&2
  exit 1
fi

if rg -n \
  'extern "c"|@(cuda|nccl|worker|worker_service|worker_protocol|core|scheduler|descriptor_file|instance_admission)\.|open_context|spawn|listen|global|pub let mut|priv let mut' \
  "$admission" "$report" --glob '*.mbt'; then
  printf '%s\n' \
    'tensor-parallel capacity packages acquired forbidden runtime authority' >&2
  exit 1
fi

if ! rg -Fq 'admitted_topology' "$admission/README.mbt.md" &&
  ! rg -Fq 'admitted local device topology' "$admission/README.mbt.md"; then
  printf '%s\n' 'admitted topology evidence is not documented' >&2
  exit 1
fi

for required in \
  'declaration.host_count != 1' \
  'declaration.pipeline_stage_count != 1' \
  'claim.world_size != facts.world_size' \
  'claim.process_ordinal != actual.process_ordinal' \
  'claim.device_name != actual.device_name' \
  'claim.target != facts.target' \
  'claim.peer_rank_count != facts.world_size - 1' \
  'facts.peer_link_count != expected_peer_link_count' \
  'claim.memory_ceiling_bytes > actual.physical_bytes' \
  'aggregate.minimum_rank_available_bytes'; do
  if ! rg -Fq "$required" "$admission"; then
    printf 'tensor-parallel exact admission check is missing: %s\n' \
      "$required" >&2
    exit 1
  fi
done

if rg -n 'Repr\(|debug_inspect|device_name.*rejection|CUDA result' \
  "$report" --glob '*.mbt'; then
  printf '%s\n' 'operator rejection report may contain unbounded payload' >&2
  exit 1
fi

for required in \
  'support.cross_node: rejected' \
  'support.pipeline_parallel: rejected' \
  'aggregate.memory.minimum_rank_available_bytes' \
  'rank.\{rank.rank()}.'; do
  if ! rg -Fq "$required" "$report/render.mbt"; then
    printf 'tensor-parallel capacity report field is missing: %s\n' \
      "$required" >&2
    exit 1
  fi
done

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    printf 'tensor-parallel source exceeds file budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done < <(find "$admission" "$report" -type f \
  \( -name '*.mbt' -o -name 'README.mbt.md' \) | sort)

printf '%s\n' \
  'LunaFlux tensor-parallel admission and report boundaries are valid.'
