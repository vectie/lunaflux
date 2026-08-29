#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() {
  printf '%s\n' "token-step performance boundary failed: $1" >&2
  exit 1
}

protocol_owner=engine/worker_protocol/plan_buffer_owner.mbt
protocol_tracker=engine/worker_protocol/plan_identity_tracker.mbt
plan_encode=engine/worker_wire/plan_encode.mbt
plan_validate=engine/worker_wire/plan_validate.mbt
root_exchange=engine/worker_process/root_bound_exchange.mbt
legacy_exchange=engine/worker_process/supervisor.mbt
rank_exchange=engine/rank_group_process/exchange.mbt
greedy_completion=engine/device_step/paged_executor_greedy_sampling.mbt
materializer=scripts/materialize-approved-tiny-bf16-launch.sh

if rg -n 'fn SchedulePlanBuffer::has_(request|slot)\(' "$protocol_owner"; then
  fail 'linear draft identity lookup returned'
fi
for anchor in identity_tracker_generation request_identity_generations \
  completion_slot_generations track_row_identity; do
  rg -Fq "$anchor" "$protocol_tracker" \
    engine/worker_protocol/plan_buffer_types.mbt ||
    fail "generation-stamped identity anchor is missing: $anchor"
done
if rg -n 'Array(::|\[)|\.push\(' "$protocol_tracker"; then
  fail 'identity lookup introduced a growing collection'
fi

validate_body="$(sed -n \
  '/fn SchedulePlanBuffer::validate_contents(/,/^}/p' "$protocol_owner")"
[ -n "$validate_body" ] || fail 'trusted submission validator is missing'
if printf '%s\n' "$validate_body" | rg -q 'for |while |token_at|prefill_row|decode_row'; then
  fail 'trusted submission restored a proportional plan rescan'
fi
[ "$(rg -F -c 'preflight_plan_for_encode(' "$plan_encode")" -eq 2 ] ||
  fail 'producer wire preflight is no longer exactly one call plus definition'
for anchor in validate_plan_frame_source frame_checksum validate_unique_scalar_identities; do
  rg -Fq "$anchor" "$plan_validate" ||
    fail "untrusted receiver validation is missing: $anchor"
done

if rg -n 'exchange_(request_ids|request_generations|expected_kinds|processed_tokens|slots|entry_count)' \
  engine/worker_process --glob '*.mbt' --glob '!*test.mbt'; then
  fail 'capacity-wide completion snapshot storage returned'
fi
for source in "$root_exchange" "$legacy_exchange" "$rank_exchange"; do
  if rg -Fq 'frame.copy_to(' "$source"; then
    fail "encoded plan memcpy returned in $source"
  fi
done
rg -Fq 'frame.validate_for(plan)' "$root_exchange" ||
  fail 'completion lost exact retained-plan validation'
rg -Fq 'seal_rank_group_wire_embedded_payload(' "$rank_exchange" ||
  fail 'rank submission no longer seals its embedded payload in place'

[ "$(rg -F -c 'copy_paged_cuda_greedy_results(' "$greedy_completion")" -eq 1 ] &&
  [ "$(rg -F -c 'copy_paged_cuda_greedy_results(' \
    engine/device_step/paged_executor_completion.mbt)" -eq 1 ] &&
  [ "$(rg -F -c 'copy_paged_cuda_greedy_results(' \
    engine/device_step/paged_executor_wire_completion.mbt)" -eq 1 ] ||
  fail 'a completion route lost its single batched device-greedy readback'
greedy_copy_body="$(sed -n \
  '/fn copy_paged_cuda_greedy_results(/,/^}/p' "$greedy_completion")"
[ "$(printf '%s\n' "$greedy_copy_body" | rg -F -c 'copy_to_fixed_host(')" -eq 1 ] ||
  fail 'device-greedy completion no longer performs one batched result readback'

for anchor in \
  'authenticated_embedded_greedy_sampling' \
  'descriptor_schema=lunaflux.runtime.v4' \
  'sampling_runtime_json=' \
  'embedded_cuda_greedy_v1' \
  'descriptor_schema=lunaflux.runtime.v3'; do
  rg -Fq "$anchor" "$materializer" ||
    fail "materialized greedy default is missing: $anchor"
done
rg -Fq 'None => @worker_wire.HostSamplingRuntime' \
  runtime/descriptor_file/schema_sections.mbt ||
  fail 'legacy descriptors no longer retain host-sampling compatibility'

printf '%s\n' 'LunaFlux token-step scan/copy/readback boundary gate passed.'
