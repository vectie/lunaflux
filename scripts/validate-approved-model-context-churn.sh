#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

runtime_types=ops/runtime_instance/spawned_context_churn_types.mbt
runtime_check=ops/runtime_instance/spawned_context_churn_check.mbt
runtime_validation=ops/runtime_instance/spawned_context_churn_validation.mbt
runtime_direct=ops/runtime_instance/spawned_direct_validation.mbt
runtime_test=ops/runtime_instance/spawned_context_churn_wbtest.mbt
campaign=tests/approved_model_spawned_physical/context_churn_campaign.mbt
request=tests/approved_model_spawned_physical/context_churn_request.mbt
campaign_test=tests/approved_model_spawned_physical/context_churn_campaign_wbtest.mbt
main=tests/approved_model_spawned_physical/main.mbt
runtime_mbti=ops/runtime_instance/pkg.generated.mbti

for file in \
  "$runtime_types" \
  "$runtime_check" \
  "$runtime_validation" \
  "$runtime_direct" \
  "$runtime_test" \
  "$campaign" \
  "$request" \
  "$campaign_test"; do
  [ -f "$file" ] || fail "context churn source is missing: $file"
done

for boundary in \
  'pub fn RuntimeInstanceOwner::validate_spawned_context_churn(Self, RuntimeSpawnedContextChurnContract) -> RuntimeSpawnedContextChurnResult raise RuntimeSpawnedContextChurnError' \
  'pub fn RuntimeSpawnedContextChurnResult::request_count(Self) -> Int' \
  'pub fn RuntimeSpawnedContextChurnResult::event_count(Self) -> Int' \
  'pub fn RuntimeSpawnedContextChurnResult::plan_count(Self) -> Int' \
  'pub fn RuntimeSpawnedContextChurnResult::final_kv_pages_free(Self) -> Int'; do
  rg -Fq "$boundary" "$runtime_mbti" ||
    fail "context churn public boundary is missing: $boundary"
done

for anchor in \
  'self.require_spawned_physical_handoff()' \
  'self.preparations[0].take_ready()' \
  'service.open_semantic_stream()' \
  'offer_spawned_validation_frame(stream, frame, turn_limit)' \
  'next_spawned_validation_event(stream, turn_limit)' \
  'close_spawned_validation_owner(self, contract.turn_limit)'; do
  rg -Fq "$anchor" "$runtime_validation" ||
    fail "context churn production-owner anchor is missing: $anchor"
done

for anchor in \
  'service.luna_telemetry()' \
  'telemetry.kv_pages_used()' \
  'telemetry.kv_pages_free()' \
  'telemetry.prefix_lookups()' \
  'sample.kv_pages_free == initial_kv_pages_free' \
  'sample.prefix_lookups == 0UL' \
  'sample.prefix_tokens_computed == expected_computed_tokens' \
  'sample.prefix_entries == 0' \
  'sample.prefix_pages == 0'; do
  rg -Fq "$anchor" "$runtime_direct" ||
    fail "context churn exact-balance anchor is missing: $anchor"
done

for anchor in \
  'owner.services.length() == 1' \
  'owner.servers.length() == 0' \
  'owner.openai_servers.length() == 0'; do
  rg -Fq "$anchor" "$runtime_direct" ||
    fail "context churn zero-network ownership anchor is missing: $anchor"
done

for anchor in \
  'CONTEXT_CHURN_FIRST_CEILING : Int = 8' \
  'CONTEXT_CHURN_SECOND_CEILING : Int = 32' \
  'CONTEXT_CHURN_THIRD_CEILING : Int = CAMPAIGN_CONTEXT_TOKENS' \
  'CONTEXT_CHURN_REQUESTS_PER_CHILD : Int = 3' \
  'CONTEXT_CHURN_FRESH_CHILD_CYCLES : Int = 3' \
  'input=@inference.Input::from_token_ids([1], limits)' \
  'cache=@inference.CachePolicy::new(' \
  'Disabled,'; do
  rg -Fq "$anchor" "$request" ||
    fail "context churn request matrix anchor is missing: $anchor"
done

for anchor in \
  '"context-churn"' \
  'run_context_churn_campaign(deployment_argument, expected_worker_sha256)'; do
  rg -Fq "$anchor" "$main" ||
    fail "context churn command route is missing: $anchor"
done

for anchor in \
  'campaign_scope=qualification-only' \
  'release_promotion=not-claimed' \
  'broader_context_length_coverage=not-established' \
  'selected_logit_correctness=not-observed' \
  'gpu_allocator_instrumentation=not-observed-by-this-mode' \
  'independent_expected_tokens=1031,2185' \
  'actual_context_tokens_per_request=3' \
  'declared_context_ceiling_matrix=8,32,256' \
  'network_observation=structural-no-listener-owner' \
  'listener_owner_count=0' \
  'network_accepts=0' \
  'children_closed=' \
  'cleanup_complete=1'; do
  rg -Fq "$anchor" "$campaign" ||
    fail "context churn evidence scope is missing: $anchor"
done

if rg -n \
  '@cuda\.|@device_worker|@worker_process|@worker_service|@online_tcp|@http1|nvrtc|PTX|JIT|Python|mock|fake' \
  "$runtime_types" "$runtime_check" "$runtime_validation" "$runtime_direct" \
  "$request" "$campaign"; then
  fail 'context churn campaign bypasses the authenticated service owner or gained excluded authority'
fi

if rg -n \
  'release_promotion=pass|broader_context_length_coverage=pass|selected_logit_correctness=pass|gpu_allocator_instrumentation=pass' \
  "$campaign"; then
  fail 'context churn campaign broadened its qualification-only evidence'
fi

for file in \
  "$runtime_types" \
  "$runtime_check" \
  "$runtime_validation" \
  "$runtime_direct" \
  "$runtime_test" \
  "$campaign" \
  "$request" \
  "$campaign_test"; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "context churn file exceeds budget: $file ($lines)"
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi
printf '%s\n' 'approved model context churn source boundary passed'
