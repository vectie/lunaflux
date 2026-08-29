#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

runtime_types=ops/runtime_instance/spawned_long_context_types.mbt
runtime_check=ops/runtime_instance/spawned_long_context_check.mbt
runtime_validation=ops/runtime_instance/spawned_long_context_validation.mbt
runtime_test=ops/runtime_instance/spawned_long_context_wbtest.mbt
runtime_direct=ops/runtime_instance/spawned_direct_validation.mbt
request=tests/approved_model_spawned_physical/long_context_request.mbt
campaign=tests/approved_model_spawned_physical/long_context_campaign.mbt
campaign_test=tests/approved_model_spawned_physical/long_context_campaign_wbtest.mbt
reference=tests/reference_corpus/prefix_referee.json
reference_test=tests/reference_corpus/reference_corpus_test.mbt
runtime_mbti=ops/runtime_instance/pkg.generated.mbti
main=tests/approved_model_spawned_physical/main.mbt

for file in \
  "$runtime_types" "$runtime_check" "$runtime_validation" "$runtime_test" \
  "$runtime_direct" "$request" "$campaign" "$campaign_test" \
  "$reference" "$reference_test"; do
  [ -f "$file" ] || fail "actual long-context source is missing: $file"
done

for boundary in \
  'pub fn RuntimeInstanceOwner::validate_spawned_long_context(Self, RuntimeSpawnedLongContextContract) -> RuntimeSpawnedLongContextResult raise RuntimeSpawnedLongContextError' \
  'pub fn RuntimeSpawnedLongContextResult::first_token(Self) -> Int' \
  'pub fn RuntimeSpawnedLongContextResult::second_token(Self) -> Int' \
  'pub fn RuntimeSpawnedLongContextResult::plan_count(Self) -> Int' \
  'pub fn RuntimeSpawnedLongContextResult::input_tokens_processed(Self) -> Int'; do
  rg -Fq "$boundary" "$runtime_mbti" ||
    fail "actual long-context public boundary is missing: $boundary"
done

for anchor in \
  'self.require_spawned_physical_handoff()' \
  'self.preparations[0].take_ready()' \
  'service.open_semantic_stream()' \
  'offer_spawned_validation_frame(stream, frame, turn_limit)' \
  'next_spawned_validation_event(stream, turn_limit)' \
  'retire_spawned_direct_stream(service, stream, contract.turn_limit)' \
  'close_spawned_validation_owner(self, contract.turn_limit)'; do
  rg -Fq "$anchor" "$runtime_validation" ||
    fail "actual long-context production-owner anchor is missing: $anchor"
done

for anchor in \
  'plan_sequence == previous_plan_sequence + input_tokens.to_uint64()' \
  'expected_live_pages' \
  'first.1 != 8UL' \
  'second.1 != 17UL' \
  'spawned_direct_resources_valid(after_first, initial.kv_pages_free, 8UL)' \
  'spawned_direct_resources_valid(terminal, initial.kv_pages_free, 17UL)'; do
  if ! rg -Fq "$anchor" "$runtime_check" "$runtime_validation"; then
    fail "actual long-context plan/KV anchor is missing: $anchor"
  fi
done

for anchor in \
  'prefix_campaign_first_input()' \
  'prefix_campaign_second_input()' \
  'PREFIX_REFEREE_FIRST_TOKEN' \
  'PREFIX_REFEREE_SECOND_TOKEN' \
  'Disabled'; do
  rg -Fq "$anchor" "$request" ||
    fail "actual long-context referee request anchor is missing: $anchor"
done

for anchor in \
  '"schema": "lunaflux.approved-bf16-prefix-referee.v1"' \
  '"input_ids": [1, 229, 153, 132, 75, 104, 111, 111]' \
  '"greedy_token": 1355' \
  '"input_ids": [1, 229, 153, 132, 75, 104, 111, 111, 114]' \
  '"greedy_token": 1240'; do
  rg -Fq "$anchor" "$reference" ||
    fail "actual long-context pinned referee changed: $anchor"
done

for anchor in \
  'prefix campaign referee is digest pinned and independently replayable' \
  '@reference.execute(' \
  'assert_eq(execution.selection().token_id(), expected_tokens[index])'; do
  rg -Fq "$anchor" "$reference_test" ||
    fail "actual long-context independent replay evidence is missing: $anchor"
done

for anchor in \
  '"long-context"' \
  'run_long_context_campaign(deployment_argument, expected_worker_sha256)'; do
  rg -Fq "$anchor" "$main" ||
    fail "actual long-context command route is missing: $anchor"
done

for anchor in \
  'campaign_scope=qualification-only' \
  'actual_input_token_lengths=8,9' \
  'actual_context_tokens_including_output=9,10' \
  'physical_page_span=1,2' \
  'beyond_9_token_input=not-established' \
  '128_or_255_token_execution=not-admitted' \
  'full_256_token_context=not-established' \
  'referee_path=approved_scalar_bf16_reference_executor' \
  'independent_expected_tokens=1355,1240' \
  'cache_permission=disabled' \
  'prefix_reuse=not-exercised' \
  'prefill_plans_per_cycle=17' \
  'plan_sequence_range_per_cycle=1-17' \
  'network_observation=structural-no-listener-owner' \
  'children_closed=' \
  'cleanup_complete=1'; do
  rg -Fq "$anchor" "$campaign" ||
    fail "actual long-context evidence boundary is missing: $anchor"
done

if rg -n \
  '@cuda\.|@device_worker|@worker_process|@worker_service|@online_tcp|@http1|nvrtc|PTX|JIT|Python|mock|fake' \
  "$runtime_types" "$runtime_check" "$runtime_validation" "$runtime_direct" \
  "$request" "$campaign"; then
  fail 'actual long-context campaign bypasses production ownership or gained excluded authority'
fi

if rg -n \
  'beyond_9_token_input=pass|128_or_255_token_execution=pass|full_256_token_context=pass|release_promotion=pass' \
  "$campaign"; then
  fail 'actual long-context campaign broadened beyond admitted evidence'
fi

for file in \
  "$runtime_types" "$runtime_check" "$runtime_validation" "$runtime_test" \
  "$runtime_direct" "$request" "$campaign" "$campaign_test"; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "actual long-context file exceeds budget: $file ($lines)"
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi
printf '%s\n' 'approved model actual long-context source boundary passed'
