#!/usr/bin/env bash
set -eu

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

runtime_mbti='ops/runtime_instance/pkg.generated.mbti'
owner='ops/runtime_instance/spawned_broad_serving.mbt'
pool='ops/runtime_instance/spawned_broad_serving_pool.mbt'
malformed='ops/runtime_instance/spawned_broad_serving_malformed.mbt'
io='ops/runtime_instance/spawned_broad_serving_io.mbt'
shared_io='ops/runtime_instance/spawned_serving_io.mbt'
checks='ops/runtime_instance/spawned_broad_serving_check.mbt'
types='ops/runtime_instance/spawned_broad_serving_types.mbt'
campaign='tests/approved_model_spawned_physical/broad_serving_campaign.mbt'
request='tests/approved_model_spawned_physical/request.mbt'
main='tests/approved_model_spawned_physical/main.mbt'
tests='ops/runtime_instance/spawned_broad_serving_wbtest.mbt'
pool_tests='ops/runtime_instance/spawned_broad_serving_pool_wbtest.mbt'
malformed_tests='ops/runtime_instance/spawned_broad_serving_malformed_wbtest.mbt'
shared_io_tests='ops/runtime_instance/spawned_serving_io_wbtest.mbt'
campaign_tests='tests/approved_model_spawned_physical/campaign_wbtest.mbt'
runner='scripts/run-approved-model-current-source-physical.sh'

for required in \
  'pub async fn RuntimeInstanceOwner::validate_spawned_broad_serving' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::request_count' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::ingress_owner' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::completion_count' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::cancellation_count' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::backpressure_count' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::network_accepts' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::network_disconnects' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::network_rejections' \
  'pub fn RuntimeSpawnedBroadServingValidationResult::kv_pages_free'; do
  if ! rg -Fq "$required" "$runtime_mbti"; then
    fail "broad BF16 opaque public seam lost: $required"
  fi
done

for typed_owner in \
  'pub(all) enum RuntimeSpawnedBroadServingIngressOwner' \
  '  BroadServingSingletonServer' \
  '  BroadServingFramedConnectionPool' \
  'pub fn RuntimeSpawnedBroadServingIngressOwner::code'; do
  if ! rg -Fq "$typed_owner" "$runtime_mbti"; then
    fail "broad BF16 typed ingress owner lost: $typed_owner"
  fi
done

# Failure-only diagnostics are a public, payload-safe qualification boundary.
# Keep the generated interface and source inventory in lockstep, and reject the
# former coarse bucket so a stale mbti cannot hide a source/interface drift.
while IFS='|' read -r variant code; do
  if ! rg -Fxq "  $variant" "$types" ||
    ! rg -Fxq "  $variant" "$runtime_mbti"; then
    fail "broad BF16 diagnostic boundary lost: $variant"
  fi
  if ! rg -Fq "\"$code\"" "$types" ||
    ! rg -Fq "\"$code\"" "$tests" "$malformed_tests"; then
    fail "broad BF16 payload-free diagnostic code lost: $code"
  fi
done <<'EOF'
BroadServingPairAccepted0|events_pair_accepted_0
BroadServingPairAccepted1|events_pair_accepted_1
BroadServingPairRemainingRead|events_pair_remaining_read
BroadServingPairRemainingAcceptedShape|events_pair_remaining_accepted_shape
BroadServingPairRemainingToken0Shape|events_pair_remaining_token_0_shape
BroadServingPairRemainingToken1Shape|events_pair_remaining_token_1_shape
BroadServingPairRemainingTokenPosition|events_pair_remaining_token_position
BroadServingPairRemainingUsageShape|events_pair_remaining_usage_shape
BroadServingPairRemainingCompletedShape|events_pair_remaining_completed_shape
BroadServingPairRemainingUnexpectedEvent|events_pair_remaining_unexpected_event
BroadServingPairRemainingCount|events_pair_remaining_count
BroadServingRecoveryAccepted|events_recovery_accepted
BroadServingRecoveryToken0|events_recovery_token_0
BroadServingRecoveryToken1|events_recovery_token_1
BroadServingRecoveryUsage|events_recovery_usage
BroadServingRecoveryCompleted|events_recovery_completed
BroadServingMalformedTransition|malformed_transition
BroadServingMalformedUnexpectedOwnerFailure|malformed_unexpected_owner_failure
BroadServingMalformedRequestMetrics|malformed_metrics_request
BroadServingMalformedNetworkMetrics|malformed_metrics_network
BroadServingMalformedResourceMetrics|malformed_metrics_resources
BroadServingMalformedReuseAccepted|malformed_reuse_events_accepted
BroadServingMalformedReuseToken0|malformed_reuse_events_token_0
BroadServingMalformedReuseToken1|malformed_reuse_events_token_1
BroadServingMalformedReuseUsage|malformed_reuse_events_usage
BroadServingMalformedReuseCompleted|malformed_reuse_events_completed
BroadServingMalformedReuseRequestMetrics|malformed_reuse_metrics_request
BroadServingMalformedReuseNetworkMetrics|malformed_reuse_metrics_network
BroadServingMalformedReuseResourceMetrics|malformed_reuse_metrics_resources
EOF

if rg -Fq 'BroadServingEvents' "$types" "$runtime_mbti" ||
  ! rg -Fq 'broad serving event checkpoint codes are stable and payload free' \
    "$tests"; then
  fail 'broad BF16 diagnostic boundary is stale or lacks payload-free tests'
fi

if rg -Fxq '  BroadServingMalformed' "$types" "$runtime_mbti" ||
  ! rg -Fq 'broad serving malformed diagnostics are stable and payload free' \
    "$malformed_tests"; then
  fail 'broad BF16 malformed diagnostics are stale or untested'
fi

awk '
  /async fn read_spawned_serving_event/ { in_helper = 1; next }
  in_helper && /client\.read/ && read_line == 0 { read_line = NR }
  in_helper && /owner\.progress/ && progress_line == 0 { progress_line = NR }
  in_helper && /^}/ { in_helper = 0 }
  END {
    if (read_line == 0 || progress_line == 0 || read_line >= progress_line) {
      exit 1
    }
  }
' "$shared_io" ||
  fail 'spawned serving reader no longer polls buffered bytes before progress'

if ! rg -Fq \
  'spawned serving reads buffered terminal before owner progress and drain' \
  "$shared_io_tests"; then
  fail 'spawned serving buffered-terminal read-before-progress regression lost'
fi

for crossing in \
  'bind_spawned_serving_listener(owner)' \
  'owner.servers[0].request_cancel()' \
  'connect_broad_serving_client' \
  'owner.progress()' \
  'drain_spawned_serving_owner' \
  'self.cleanup_complete()'; do
  if ! rg -Fq "$crossing" "$owner" "$pool" "$io"; then
    fail "broad BF16 owner crossing lost: $crossing"
  fi
done

for metric in \
  'Admissions' 'Completions' 'Cancellations' 'Deadlines' 'Failures' \
  'PromptTokens' 'GeneratedTokens' 'WorkerFailures' 'Backpressure' \
  'NetworkAccepts' 'NetworkDisconnects' 'NetworkRejections' \
  'QueueDepth' 'ActiveRequests' 'KvPagesUsed' 'KvPagesFree'; do
  if ! rg -Fq "$metric" "$checks"; then
    fail "broad BF16 exact balance lost: $metric"
  fi
done

for case_gate in \
  'saw_two_live' \
  'BroadServingCancellation' \
  'run_broad_serving_foreign_rejection' \
  'run_broad_serving_recovery' \
  'run_broad_serving_malformed_fail_closed' \
  'run_broad_serving_post_malformed_recovery' \
  'owner.phase() != Ready' \
  'owner.spawned_serving_ingress_listening()' \
  'expected_accepts = if pooled { 7UL } else { 5UL }' \
  'event_count: if pooled' \
  'ingress_owner: if pooled' \
  'await_broad_serving_pool_connections(owner, 2, contract.turn_limit)' \
  'cancelled.close()'; do
  if ! rg -Fq "$case_gate" "$owner" "$pool" "$malformed"; then
    fail "broad BF16 physical case lost: $case_gate"
  fi
done

if ! rg -Fq '[_, "broad-serving", deployment_argument, expected_worker_sha256]' \
    "$main" ||
  ! rg -Fq 'owner.validate_spawned_broad_serving(contract)' "$campaign" ||
  ! rg -Fq 'broad_campaign_validation_contract' "$request" ||
  ! rg -Fq 'lunaflux-approved-model-broad-bf16-serving-qualification.v2' \
    "$runner"; then
  fail 'broad BF16 current-source runner integration is incomplete'
fi

if rg -n \
  '@online_tcp|@online_session|@socket|@worker_service|@worker_process|@device_worker|@cuda\.' \
  "$campaign" "$request" "$main"; then
  fail 'broad BF16 campaign bypasses the opaque runtime owner'
fi

for evidence in \
  'campaign_scope=qualification-only' \
  'production_readiness=not-claimed' \
  'performance_baseline=not-established' \
  'benchmark_claim=not-made' \
  'mixed_concurrent_requests=pass' \
  'ingress_owner=singleton-server' \
  'ingress_owner=framed-connection-pool' \
  'saturation_proof=scheduler-backpressure' \
  'saturation_proof=connection-capacity-deferral' \
  'saturation_backpressure=pass' \
  'saturation_backpressure=not-claimed' \
  'connection_capacity_deferral=not-exercised-singleton-server' \
  'connection_capacity_deferral=pass' \
  'overload_rejection=not-exercised-single-connection-owner' \
  'overload_rejection=not-exercised-capacity-deferral' \
  'cancellation_isolation=pass' \
  'typed_foreign_model_rejection=pass' \
  'malformed_native_frame=fail-closed-connection-isolation-pass' \
  'malformed_http=not-exercised-native-framed-owner' \
  'post_malformed_same_owner_recovery=pass' \
  'fresh_owner_restart=pass' \
  'child_closed=2'; do
  if ! rg -Fq "$evidence" "$campaign"; then
    fail "broad BF16 evidence lost scoped claim: $evidence"
  fi
done

if rg -n \
  'production_readiness=pass|performance_baseline=pass|benchmark_claim=pass|malformed_http=pass|overload_rejection=pass' \
  "$campaign" ||
  ! rg -Fq 'broad serving malformed retirement and reuse balances are exact' \
    "$malformed_tests" ||
  ! rg -Fq 'broad serving contract copies every authenticated request frame' \
    "$tests" ||
  ! rg -Fq 'broad BF16 campaign is qualification-only and names unexercised cases' \
    "$campaign_tests"; then
  fail 'broad BF16 hostile tests or claim exclusions are incomplete'
fi

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "broad BF16 source exceeds file budget: $file ($lines)"
  fi
done < <(printf '%s\n' \
  "$owner" "$malformed" "$io" "$shared_io" "$checks" "$types" \
  "$pool" \
  "$campaign" "$tests" "$pool_tests" "$malformed_tests" \
  "$shared_io_tests" | sort)

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'approved-model broad BF16 serving qualification gate passed'
