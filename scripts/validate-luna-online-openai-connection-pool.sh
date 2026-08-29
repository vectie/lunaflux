#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon check service/online_session service/online_tcp \
  --target native --release --deny-warn --warn-list +73
moon test service/online_session service/online_tcp \
  --target native --release --deny-warn --warn-list +73
moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73

mbti="service/online_tcp/pkg.generated.mbti"
for type in \
  LunaOnlineOpenAIConnectionPool \
  LunaOnlineOpenAIConnectionPoolPreparation \
  LunaOnlineOpenAIConnectionPoolLimits; do
  if ! rg -q --pcre2 -U \
      "pub struct ${type} \\{\\s*// private fields\\s*\\}" "$mbti"; then
    printf 'OpenAI connection-pool authority is not opaque: %s\n' "$type" >&2
    exit 1
  fi
done

# Bind opens one semantic stream for the entire pool. Accept/reuse never opens
# another scheduler or framed-service owner.
if [ "$(rg -o 'open_semantic_stream\(' \
    service/online_tcp/openai_pool_*.mbt | wc -l | tr -d ' ')" -ne 1 ] ||
  [ "$(rg -o 'take_open_stream\(' \
    service/online_tcp/openai_pool_*.mbt | wc -l | tr -d ' ')" -ne 1 ] ||
  rg -n 'prepare_owned|Scheduler|WorkerServiceBinding|begin_multi|take_event' \
    service/online_tcp/openai_pool_*.mbt; then
  printf '%s\n' 'OpenAI pool duplicated scheduling or stream authority' >&2
  exit 1
fi

# All connection workspaces, route/credit slots, socket holders, input bytes,
# and one whole-event output cell are startup-created. Warm progression may
# mutate only those cells; constructors and collection-capacity growth stay out.
if ! rg -q 'FixedArray::makei\(pool_limits.connection_capacity' \
    service/online_tcp/openai_pool_prepare.mbt ||
  ! rg -q 'output: LunaOnlineTcpOutputScratch::new' \
    service/online_tcp/openai_pool_prepare.mbt ||
  rg -n 'Array::new|FixedArray::make|Bytes::make|StringBuilder|Buffer::new' \
    service/online_tcp/openai_pool_accept.mbt \
    service/online_tcp/openai_pool_ingress.mbt \
    service/online_tcp/openai_pool_output.mbt \
    service/online_tcp/openai_pool_progress.mbt \
    service/online_tcp/openai_pool_lifecycle.mbt; then
  printf '%s\n' 'OpenAI pool warmed path gained dynamic construction' >&2
  exit 1
fi

# A semantic event is copied fully into the bounded slot before it is ACKed;
# socket output starts only from the published immutable Flight. A full/busy
# slot cancels and discards the exact route instead of pinning global progress.
if ! rg -q --pcre2 -U \
    'copy_compat_view[\s\S]*copy_openai_response_at[\s\S]*semantic_events\[0\]\.delivered\(\)[\s\S]*publish_output' \
    service/online_tcp/openai_pool_output.mbt ||
  ! rg -q --pcre2 -U \
    '!slot\.output\.is_idle\(\)[\s\S]*discard_semantic_event' \
    service/online_tcp/openai_pool_output.mbt ||
  ! rg -q --pcre2 -U \
    'cancel_and_discard\(\)[\s\S]*slot\.cancel_requested = true[\s\S]*begin_slot_retirement' \
    service/online_tcp/openai_pool_output.mbt ||
  ! rg -q --pcre2 -U \
    'write_slot_output[\s\S]*write_once_at' \
    service/online_tcp/openai_pool_output.mbt; then
  printf '%s\n' 'OpenAI pool copy/ACK/backpressure order drifted' >&2
  exit 1
fi

# The shared framed receipt is never interleaved. Exact opaque routes bind each
# event/rejection to one current slot generation, and only short I/O polls can
# await on the serialized owner.
if ! rg -q --pcre2 -U \
    'if self\.ingress_slot >= 0 && self\.ingress_slot != index[\s\S]*self\.ingress_slot = index' \
    service/online_tcp/openai_pool_ingress.mbt ||
  ! rg -q --pcre2 -U \
    'ingress\.request_route\(\)[\s\S]*slot\.routes\.push\(route\)[\s\S]*same_request' \
    service/online_tcp/openai_pool_ingress.mbt ||
  ! rg -q --pcre2 -U \
    'for index, slot in self\.slots[\s\S]*slot\.route\(\)\.same_request\(route\)' \
    service/online_tcp/openai_pool_lifecycle.mbt ||
  [ "$(rg -o 'with_timeout_opt' service/online_tcp/openai_pool_*.mbt | \
    wc -l | tr -d ' ')" -ne 3 ] ||
  [ "$(rg -o 'pool_limits.io_poll_timeout_millis' \
    service/online_tcp/openai_pool_{accept,ingress,output}.mbt | \
    wc -l | tr -d ' ')" -ne 3 ]; then
  printf '%s\n' 'OpenAI pool route serialization or bounded I/O drifted' >&2
  exit 1
fi

if ! rg -q -F \
    'test "Luna framed ingress route is stable, opaque, and cancellation exact" {' \
    service/online_session/coordinator_route_wbtest.mbt ||
  ! rg -q -F \
    'test "OpenAI connection pool limits are startup bounded" {' \
    service/online_tcp/openai_pool_wbtest.mbt ||
  ! rg -q 'run_luna_openai_pool_evidence\(' \
    tests/worker_service_e2e/online_session_e2e_entry.mbt; then
  printf '%s\n' 'OpenAI pool hostile/reuse/black-box evidence drifted' >&2
  exit 1
fi

e2e=tests/worker_service_e2e/luna_openai_pool_evidence.mbt
slow_line=$(rg -n '^  let slow = ' "$e2e" | cut -d: -f1)
fast_line=$(rg -n '^  let fast = ' "$e2e" | cut -d: -f1)
two_line=$(rg -n 'progress_openai_pool_to_connections\(pool, 2,' "$e2e" | head -n 1 | cut -d: -f1)
one_line=$(rg -n 'progress_openai_pool_to_connections\(pool, 1,' "$e2e" | tail -n 1 | cut -d: -f1)
reused_line=$(rg -n '^  let reused = ' "$e2e" | cut -d: -f1)
if [ -z "$slow_line" ] || [ -z "$fast_line" ] || [ -z "$two_line" ] ||
  [ -z "$one_line" ] || [ -z "$reused_line" ] ||
  [ "$slow_line" -ge "$fast_line" ] || [ "$fast_line" -ge "$two_line" ] ||
  [ "$two_line" -ge "$one_line" ] || [ "$one_line" -ge "$reused_line" ]; then
  printf '%s\n' 'OpenAI pool hostile/reuse/black-box evidence order drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux OpenAI connection-pool bounded-owner and warmed-source gate passed.'
