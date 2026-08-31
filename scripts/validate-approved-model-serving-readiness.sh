#!/usr/bin/env bash
set -eu

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

runtime_mbti='ops/runtime_instance/pkg.generated.mbti'
owner='ops/runtime_instance/spawned_serving.mbt'
io='ops/runtime_instance/spawned_serving_io.mbt'
checks='ops/runtime_instance/spawned_serving_check.mbt'
types='ops/runtime_instance/spawned_serving_types.mbt'
campaign='tests/approved_model_spawned_physical/serving_campaign.mbt'
request='tests/approved_model_spawned_physical/request.mbt'
main='tests/approved_model_spawned_physical/main.mbt'
tests='tests/approved_model_spawned_physical/campaign_wbtest.mbt'

for required in \
  'pub async fn RuntimeInstanceOwner::validate_spawned_serving_request' \
  'pub fn RuntimeSpawnedServingValidationResult::first_token' \
  'pub fn RuntimeSpawnedServingValidationResult::second_token' \
  'pub fn RuntimeSpawnedServingValidationResult::event_count' \
  'pub fn RuntimeSpawnedServingValidationResult::kv_pages_free' \
  'pub fn RuntimeSpawnedServingValidationResult::network_accepts' \
  'pub fn RuntimeSpawnedServingValidationResult::network_disconnects'; do
  if ! rg -Fq "$required" "$runtime_mbti"; then
    fail "serving-readiness opaque public seam lost: $required"
  fi
done

if ! rg -Fq 'owner.require_spawned_physical_handoff()' "$owner" ||
  ! rg -Fq 'owner.progress()' "$owner" ||
  ! rg -Fq 'owner.health()' "$owner" ||
  ! rg -Fq 'owner.readiness()' "$owner" ||
  ! rg -Fq 'RuntimeHealthy' "$owner" ||
  ! rg -Fq 'RuntimeReady' "$owner" ||
  ! rg -Fq '@socket.Tcp::connect(addr)' "$io" ||
  ! rg -Fq 'owner.singular_ingress_addr()' "$io" ||
  ! rg -Fq 'owner.spawned_serving_ingress_active()' "$owner" "$io" ||
  ! rg -Fq 'owner.spawned_serving_ingress_listening()' "$owner" ||
  ! rg -Fq 'owner.spawned_serving_terminal_ingress_live()' "$owner" ||
  ! rg -Fq 'owner.spawned_serving_terminal_disconnected()' "$owner" ||
  ! rg -Fq 'self.framed_pools[0].active_connection_count()' "$checks" ||
  ! rg -Fq '@framed_wire.EventFrameBuffer::new' "$owner" ||
  ! rg -Fq 'client.close()' "$owner"; then
  fail 'serving validator no longer crosses the opaque owner and real listener'
fi

for metric in \
  'Admissions' 'Completions' 'Cancellations' 'Deadlines' 'Failures' \
  'PromptTokens' 'GeneratedTokens' 'WorkerFailures' 'NetworkAccepts' \
  'NetworkDisconnects' 'NetworkRejections' 'QueueDepth' 'ActiveRequests' \
  'KvPagesUsed' 'KvPagesFree'; do
  if ! rg -Fq "$metric" "$checks"; then
    fail "serving validator lost exact metric balance: $metric"
  fi
done

for event in \
  'Accepted(accepted)' 'Token(token)' 'Usage(usage)' \
  'Completed(completed)' 'TokenLimit'; do
  if ! rg -Fq "$event" "$checks"; then
    fail "serving validator lost canonical event check: $event"
  fi
done

if ! rg -Fq 'owner.begin_drain()' "$owner" ||
  ! rg -Fq 'owner.cleanup_complete()' "$owner" ||
  ! rg -Fq 'spawned_serving_closed_state_valid' "$owner" ||
  ! rg -Fq 'owner.spawned_serving_ingress_listening()' "$owner";
then
  fail 'serving validator can publish before listener child or KV cleanup'
fi

if rg -n \
  'RuntimeSpawnedServingValidationResult.*(LunaOnline|Tcp|Worker|Service|Scheduler|Device|Cuda)' \
  "$runtime_mbti" ||
  rg -n '@cuda\.|@device_worker|@worker_service|@worker_process|nvrtc|PTX|JIT' \
    "$owner" "$io" "$checks" "$types"; then
  fail 'serving validation leaked or bypassed lower mutable authority'
fi

if ! rg -Fq '[_, "serving", deployment_argument, expected_worker_sha256]' \
    "$main" ||
  ! rg -Fq '@runtime_instance.prepare' \
    tests/approved_model_spawned_physical/campaign.mbt ||
  ! rg -Fq 'owner.validate_spawned_serving_request(contract)' "$campaign" ||
  ! rg -Fq 'campaign_serving_validation_contract' "$request"; then
  fail 'physical serving campaign no longer reaches the production runtime owner'
fi

if rg -n \
  '@online_tcp|@online_session|@socket|@worker_service|@worker_process|@device_worker|@cuda\.' \
  "$campaign" "$request" "$main"; then
  fail 'physical serving campaign bypasses the opaque runtime owner'
fi

for evidence in \
  'traffic_readiness_observed=1' \
  'native_tcp_listener=pass' \
  'event_order=accepted,token,token,usage,completed' \
  'request_count=1' \
  'kv_pages_used_after_request=0' \
  'listener_closed=1' \
  'child_closed=1' \
  'tls_validation=not-run' \
  'performance_validation=not-run'; do
  if ! rg -Fq "$evidence" "$campaign"; then
    fail "serving evidence lost scoped claim: $evidence"
  fi
done

if rg -n 'tls_validation=pass|performance_validation=pass' \
    "$campaign" tests/approved_model_spawned_physical/README.md ||
  ! rg -Fq 'serving campaign pins and authenticates the exact bounded frame' \
    "$tests" ||
  ! rg -Fq 'serving evidence scope excludes TLS and performance claims' \
    "$tests" ||
  ! rg -Fq 'spawned serving terminal metrics require exact request and KV balance' \
    ops/runtime_instance/spawned_serving_wbtest.mbt ||
  ! rg -Fq 'spawned serving admits exactly one native server or framed pool' \
    ops/runtime_instance/spawned_serving_wbtest.mbt ||
  ! rg -Fq 'spawned serving framed pool requires exact live connection balance' \
    ops/runtime_instance/spawned_serving_wbtest.mbt ||
  ! rg -Fq 'spawned serving terminal ingress preserves server and pool lifecycle' \
    ops/runtime_instance/spawned_serving_wbtest.mbt ||
  ! rg -Fq 'spawned serving request rejects buffered cacheable and sampled variants' \
    ops/runtime_instance/spawned_serving_wbtest.mbt ||
  ! rg -Fq 'spawned serving contract defensively copies the authenticated frame' \
    ops/runtime_instance/spawned_serving_wbtest.mbt; then
  fail 'serving hostile tests or claim exclusions are incomplete'
fi

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "serving validation source exceeds file budget: $file ($lines)"
  fi
done < <(printf '%s\n' \
  "$owner" "$io" "$checks" "$types" "$campaign" "$request" | sort)

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'approved model serving-readiness boundary gate passed'
