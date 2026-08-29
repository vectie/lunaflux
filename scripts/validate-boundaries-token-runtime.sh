#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

fail_matches() {
  description=$1
  shift
  if matches=$(rg -n "$@" 2>/dev/null); then
    printf '%s\n%s\n' "$description" "$matches" >&2
    failed=1
  fi
}

if rg -n '^pub fn TokenBuffer::token_ids\(|^pub fn TokenizedRequest::(input_tokens|input_token_at)\(' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
  rg -n --pcre2 -U \
    'pub struct (TokenBuffer|TokenizedRequest) \{\n  (?!// private fields)' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
  rg -n --pcre2 -U \
    'pub struct (TokenBuffer|TokenizedRequest) \{(?s:[^}]*)\} derive\([^)]*Debug' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
  printf '%s\n' \
    'scheduler token request leaked raw token arrays or debuggable representation' >&2
  failed=1
fi

token_bound_surface="$(rg \
  '^pub fn (LunaTokenBufferBoundStatus::|TokenBuffer::maximum_token_status\()' \
  contracts/inference/pkg.generated.mbti || true)"
if ! rg -q --pcre2 -U \
    '^pub struct LunaTokenBufferBoundStatus \{\n  // private fields\n\}$' \
    contracts/inference/pkg.generated.mbti ||
  rg -q --pcre2 -U \
    '^pub struct LunaTokenBufferBoundStatus \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti ||
  [ "$token_bound_surface" != \
  $'pub fn LunaTokenBufferBoundStatus::is_out_of_range(Self) -> Bool\npub fn LunaTokenBufferBoundStatus::is_stale(Self) -> Bool\npub fn LunaTokenBufferBoundStatus::is_within_bound(Self) -> Bool\npub fn TokenBuffer::maximum_token_status(Self, Int) -> LunaTokenBufferBoundStatus' ]; then
  printf '%s\n' \
    'authenticated Luna token-maximum bound opacity or surface drifted' >&2
  failed=1
fi

if rg -n \
    '^pub fn TokenBuffer::.*maximum.* -> Int$|^pub fn TokenizedRequest::input_maximum_token_status\(' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
  printf '%s\n' \
    'raw token maximum or scheduler forwarding authority became public' >&2
  failed=1
fi
if ! rg -F -x -q \
    'pub fn TokenizedRequest::new(request_id~ : @inference.RequestId, model_identity~ : @spec.ModelIdentity, input_tokens~ : @inference.TokenBuffer, sampling~ : @inference.SamplingParameters, sampling_seed~ : @inference.SamplingSeed, stop_tokens~ : @inference.LunaRequestStopTokenView, stream_preference~ : @inference.StreamPreference, effective_limits~ : @inference.EffectiveLimits, admission_deadline~ : @inference.AdmissionDeadline) -> Self raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn TokenizedRequest::new_with_prefix(request_id~ : @inference.RequestId, model_identity~ : @spec.ModelIdentity, input_tokens~ : @inference.TokenBuffer, sampling~ : @inference.SamplingParameters, sampling_seed~ : @inference.SamplingSeed, stop_tokens~ : @inference.LunaRequestStopTokenView, stream_preference~ : @inference.StreamPreference, effective_limits~ : @inference.EffectiveLimits, admission_deadline~ : @inference.AdmissionDeadline, tokenizer_digest~ : @tokenizer.TokenizerDigest, prefix_semantic~ : @inference.LunaRequestPrefixView) -> Self raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn PreparedAdmission::retain_stop_tokens_into(Self, @inference.LunaRequestStopTokenRetentionSlot) -> Unit raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn PreparedExclusiveAdmission::retain_stop_tokens_into(Self, @inference.LunaRequestStopTokenRetentionSlot) -> Unit raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  rg -n '^pub fn TokenizedRequest::(stops|cache|stop_tokens|semantic)' \
    scheduler/core/pkg.generated.mbti ||
  rg -n '^pub fn TokenizedRequest::.*LunaRequestPrefixView' \
    scheduler/core/pkg.generated.mbti | rg -v '::new_with_prefix' ||
  rg -n 'LunaRequestSemanticView|StopConditions|CachePolicy' \
    scheduler/core/pkg.generated.mbti; then
  printf '%s\n' 'scheduler token-only semantic authority surface drifted' >&2
  failed=1
fi

if rg -n \
    '@inference\.(LunaRequestSemanticView|StopConditions|CachePolicy)|\.stops\(\)|\.cache\(\)|stop_string_(count|length|byte_at)|cache_scope_(length|byte_at)|token_only\(' \
    scheduler/core --glob '*.mbt' --glob '!**/*_test.mbt' \
    --glob '!**/*_wbtest.mbt' ||
  ! rg -q \
    'priv stop_tokens : @inference\.LunaRequestStopTokenView' \
    scheduler/core/request.mbt ||
  ! rg -q \
    '@inference\.LunaRequestStopTokenRetentionSlot' \
    scheduler/core/owner_types.mbt ||
  ! rg -F -q \
    'retention.is_stop_token(token)' \
    scheduler/core/completion_preflight.mbt; then
  printf '%s\n' \
    'scheduler source escaped the narrow Luna stop-token projection boundary' >&2
  failed=1
fi

if rg -n \
  'prepare_owned_session|OnlineSession(::|Preparation|Progress|Cleanup|Error|Failure|Rule|\s*\{)' \
  service/online_session tests/worker_service_e2e/online_session_*.mbt; then
  printf '%s\n' 'removed one-shot OnlineSession API returned' >&2
  failed=1
fi

if rg -n \
  '^pub\(all\) struct LunaOnlineRequestTicket|^pub struct LunaOnlineRequestTicket\(|^pub fn LunaOnlineRequestTicket::(value|epoch|id|raw)\(|^pub fn LunaOnlineInstanceAdmission::(epoch|raw|status)\(' \
  service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'Luna online request tickets must remain opaque scalar authority' >&2
  failed=1
fi

persistent_lower_calls=$(rg -n \
  '\.(retire_terminal_request|shutdown_clean_empty|retry_close_empty)\(' \
  --glob '*.mbt' || true)
if [ -n "$persistent_lower_calls" ] &&
  printf '%s\n' "$persistent_lower_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'persistent worker retire/shutdown authority escaped aggregate scope' >&2
  failed=1
fi

# Production code is MoonBit plus narrow C stubs. Python may be used by neither
# the runtime nor its normal validation path.
if python_files=$(rg --files --glob '*.py' 2>/dev/null); then
  printf '%s\n%s\n' 'Python files are forbidden in the LunaFlux repository:' "$python_files" >&2
  failed=1
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
