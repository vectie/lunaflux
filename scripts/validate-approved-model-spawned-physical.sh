#!/usr/bin/env bash
set -eu

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

campaign='tests/approved_model_spawned_physical/campaign.mbt'
identity='tests/approved_model_spawned_physical/identity.mbt'
tests='tests/approved_model_spawned_physical/campaign_wbtest.mbt'
runtime_mbti='ops/runtime_instance/pkg.generated.mbti'
validation_sources=(
  ops/runtime_instance/spawned_validation.mbt
  ops/runtime_instance/spawned_validation_check.mbt
  ops/runtime_instance/spawned_validation_progress.mbt
  ops/runtime_instance/spawned_validation_types.mbt
)

for required in \
  'pub fn RuntimeInstanceOwner::require_spawned_physical_handoff(Self) -> Unit raise RuntimeInstanceError' \
  'pub fn RuntimeInstanceOwner::validate_spawned_physical_request(Self, RuntimeSpawnedValidationContract) -> RuntimeSpawnedValidationResult raise RuntimeSpawnedValidationError' \
  'pub fn RuntimeSpawnedValidationResult::first_token(Self) -> Int' \
  'pub fn RuntimeSpawnedValidationResult::second_token(Self) -> Int' \
  'pub fn RuntimeReleaseAdmission::tokenizer_sha256(Self) -> String' \
  'pub fn RuntimeReleaseAdmission::worker_executable_sha256(Self) -> String'; do
  if ! rg -Fq "$required" "$runtime_mbti"; then
    fail "spawned handoff public boundary lost: $required"
  fi
done

if ! rg -Fq 'SpawnedPhysicalHandoff' "$runtime_mbti" ||
  ! rg -q 'spawned_physical_handoff_matches\(' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q 'self\.preparations\[0\]\.state\(\) == LunaOnlineInstanceReady' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q 'phase == HealthyNotReady' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q '^  preparation_ready &&$' ops/runtime_instance/owner.mbt ||
  ! rg -q '^  preparation_count == 1 &&$' ops/runtime_instance/owner.mbt ||
  ! rg -q '^  cleanup_count == 0 &&$' ops/runtime_instance/owner.mbt ||
  ! rg -q '^  service_count == 0 &&$' ops/runtime_instance/owner.mbt ||
  ! rg -q '^  server_count == 0 &&$' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q 'borrowed_root_count == 0' ops/runtime_instance/owner.mbt; then
  fail 'spawned handoff can be forged outside the exact child-ready owner state'
fi

if ! rg -Fq \
  'spawned physical handoff rejects every synthetic owner state' \
  ops/runtime_instance/owner_wbtest.mbt ||
  ! rg -Fq 'spawned physical handoff scalar predicate is exact' \
    ops/runtime_instance/owner_wbtest.mbt; then
  fail 'spawned handoff hostile owner-state evidence is missing'
fi

if ! rg -Fq 'self.preparations[0].take_ready()' \
    ops/runtime_instance/spawned_validation.mbt ||
  ! rg -Fq 'service.open_semantic_stream()' \
    ops/runtime_instance/spawned_validation.mbt ||
  ! rg -Fq 'stream.offer_luna_framed(' \
    ops/runtime_instance/spawned_validation_progress.mbt ||
  ! rg -Fq 'stream.take_semantic_event()' \
    ops/runtime_instance/spawned_validation_progress.mbt ||
  ! rg -Fq 'service.luna_plan_telemetry()' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'spawned_validation_kv_state_valid(' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'active_requests == 1' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'kv_pages_used == 1' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'kv_pages_free == total_kv_pages - 1' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'active_requests == 0' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'kv_pages_used == 0' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'kv_pages_free == total_kv_pages' \
    ops/runtime_instance/spawned_validation_check.mbt ||
  ! rg -Fq 'service.begin_drain()' \
    ops/runtime_instance/spawned_validation.mbt ||
  ! rg -Fq 'owner.services.clear()' \
    ops/runtime_instance/spawned_validation.mbt; then
  fail 'spawned execution no longer crosses and closes the owned service path'
fi

if rg -n \
  '@cuda\.|@device_worker|@worker_process|@worker_service|@online_tcp|nvrtc|PTX|JIT|Array::new|StringBuilder' \
  "${validation_sources[@]}" ||
  rg -n \
    'RuntimeSpawnedValidationResult.*(LunaOnline|Worker|Device|Scheduler|Logit)' \
    "$runtime_mbti"; then
  fail 'spawned validator bypasses service ownership or gained dynamic/result authority'
fi

if ! rg -Fq \
  '852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30' \
  "$identity" tests/reference_corpus/MANIFEST.md ||
  ! rg -Fq \
  'fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b' \
  "$identity" tests/reference_corpus/MANIFEST.md ||
  ! rg -Fq \
  '3579d71fd57e04f5a364d824d3a2ec3e913dbb67' \
  "$identity" tests/reference_corpus/MANIFEST.md ||
  ! rg -q 'vocabulary_size=3000' "$identity" ||
  ! rg -q 'hidden_size=16' "$identity" ||
  ! rg -q 'layer_count=2' "$identity" ||
  ! rg -q 'batch=@llama\.LlamaPagedBatchEnvelope::single_row\(\)' "$identity"; then
  fail 'campaign is not pinned to the checked-in upstream tiny BF16 model graph'
fi

preflight_line=$(rg -n '@runtime_instance\.preflight_release' "$campaign" | cut -d: -f1)
identity_line=$(rg -n '^  require_pinned_release\(' "$campaign" | cut -d: -f1)
prepare_line=$(rg -n 'let owner = @runtime_instance\.prepare' "$campaign" | cut -d: -f1)
handoff_line=$(rg -n 'owner\.require_spawned_physical_handoff' "$campaign" | cut -d: -f1)
execute_line=$(rg -n '^  let result = owner\.validate_spawned_physical_request' "$campaign" | cut -d: -f1)
drain_line=$(rg -n '^  let drained = drain_campaign_owner' "$campaign" | cut -d: -f1)
publish_line=$(rg -n '^  println\(evidence\)' "$campaign" | cut -d: -f1)
if [ -z "$preflight_line" ] || [ -z "$identity_line" ] ||
  [ -z "$prepare_line" ] || [ -z "$handoff_line" ] ||
  [ -z "$execute_line" ] || [ -z "$drain_line" ] || [ -z "$publish_line" ] ||
  [ "$preflight_line" -ge "$identity_line" ] ||
  [ "$identity_line" -ge "$prepare_line" ] ||
  [ "$prepare_line" -ge "$handoff_line" ] ||
  [ "$handoff_line" -ge "$execute_line" ] ||
  [ "$execute_line" -ge "$drain_line" ] ||
  [ "$drain_line" -ge "$publish_line" ]; then
  fail 'campaign no longer preflights, pins, prepares, proves, cleans, then publishes'
fi

if ! rg -Fq 'expected_worker.as_hex()' "$campaign" ||
  ! rg -Fq 'owner.readiness() == RuntimeNotReady' "$campaign" ||
  ! rg -Fq 'traffic_readiness=0' "$campaign" ||
  ! rg -Fq 'plan_executed=2' "$campaign" ||
  ! rg -Fq 'generated_tokens=' "$campaign" ||
  ! rg -Fq 'selected_logit_correctness=not-observed' "$campaign" ||
  rg -n 'traffic_readiness=1|selected_logit_correctness=pass' \
    "$campaign" tests/approved_model_spawned_physical/README.md; then
  fail 'campaign lost execution evidence or broadened into serving/logit claims'
fi

if rg -n \
  'fake|mock|@cuda\.|@device_worker|@worker_service|@worker_process|@online_tcp|kernel_release|shape_matrix|nvrtc|Python|JIT' \
  tests/approved_model_spawned_physical --glob '*.mbt' --glob 'moon.pkg' ||
  rg -n '@runtime_instance\.progress|\.progress\(' "$campaign" | rg -v 'owner\.progress\(\)'; then
  fail 'campaign bypasses the opaque production owner or gained compiler/listener authority'
fi

for hostile in \
  'pinned campaign rejects recipe model tokenizer and child substitution' \
  'dense_llama_i8_paged_aot_v6' \
  '"b".repeat(64)' \
  '"c".repeat(64)' \
  '"d".repeat(64)' \
  '"e".repeat(64)' \
  'traffic_readiness=0' \
  'plan_executed=2' \
  'context_capacity_tokens=256' \
  'emergency_decode_page_reserve=positive-admitted' \
  'selected_logit_correctness=not-observed'; do
  if ! rg -Fq "$hostile" "$tests"; then
    fail "campaign hostile evidence lost anchor: $hostile"
  fi
done

for geometry in \
  'CAMPAIGN_TOKENS_PER_PAGE : Int = 8' \
  'CAMPAIGN_CONTEXT_PAGES : Int = 32' \
  'CAMPAIGN_CONTEXT_TOKENS : Int = CAMPAIGN_TOKENS_PER_PAGE *' \
  'CAMPAIGN_CONTEXT_PAGES' \
  'max_context_tokens=CAMPAIGN_CONTEXT_TOKENS' \
  'context_ceiling=CAMPAIGN_CONTEXT_TOKENS'; do
  if ! rg -Fq "$geometry" tests/approved_model_spawned_physical/request.mbt; then
    fail "campaign context geometry lost anchor: $geometry"
  fi
done

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "spawned physical campaign source exceeds file budget: $file ($lines)"
  fi
done < <(find tests/approved_model_spawned_physical -name '*.mbt' -type f | sort)

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "spawned validation source exceeds file budget: $file ($lines)"
  fi
done < <(printf '%s\n' "${validation_sources[@]}" | sort)

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'approved tiny-model spawned physical execution gate passed'
