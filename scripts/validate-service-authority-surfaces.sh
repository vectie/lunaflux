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

for required_authority_interface in \
  contracts/inference/pkg.generated.mbti \
  model/spec/pkg.generated.mbti \
  scheduler/core/pkg.generated.mbti; do
  if [ ! -f "$required_authority_interface" ]; then
    printf '%s\n' \
      "required authority interface ${required_authority_interface} is missing" >&2
    failed=1
  fi
done

. scripts/authority-boundary-helpers.sh

for request_authority in \
    AdmissionDeadline \
    CachePolicy \
    CacheScope \
    DeadlineBudget \
    EffectiveLimits \
    GenerateRequest \
    InferenceLimits \
    ProtocolVersion \
    RequestId \
    SamplingParameters \
    SamplingSeed \
    StopConditions \
    TextInput \
    TokenBuffer \
    TraceCorrelation; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${request_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference request authority ${request_authority} is not opaque" >&2
    failed=1
  fi
done
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti AdmissionDeadline from_receipt
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti CachePolicy new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti CacheScope from_ascii
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti DeadlineBudget from_milliseconds
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti EffectiveLimits new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti GenerateRequest new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti InferenceLimits new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti Input 'from_token_ids
from_utf8'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti ProtocolVersion 'from_wire
v1'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti RequestId new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti SamplingParameters 'greedy
sample'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti SamplingSeed new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti StopConditions 'new
token_only'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TextInput from_utf8
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TokenBuffer new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TraceCorrelation from_ascii
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti AdmissionDeadline AdmissionDeadline::from_receipt
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti CachePolicy 'CachePolicy::new
GenerateRequest::cache'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti CacheScope 'CachePolicy::scope
CacheScope::from_ascii'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti DeadlineBudget 'DeadlineBudget::from_milliseconds
GenerateRequest::deadline
LunaTextRequestHandoffLease::deadline'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti EffectiveLimits 'AcceptedEvent::effective_limits
EffectiveLimits::new
LunaTextRequestHandoffLease::effective_limits'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti GenerateRequest GenerateRequest::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti InferenceLimits 'InferenceLimits::new
LunaRequestSemanticView::inference_limits
LunaTextRequestHandoffLease::inference_limits'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti Input 'GenerateRequest::input
Input::from_token_ids
Input::from_utf8'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti ProtocolVersion 'GenerateRequest::protocol_version
ProtocolVersion::from_wire
ProtocolVersion::v1'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti RequestId 'GenerateRequest::request_id
LunaTextRequestHandoffLease::request_id
RequestId::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti SamplingParameters 'GenerateRequest::sampling
LunaValidatedSampling::materialize
SamplingParameters::greedy
SamplingParameters::sample'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti SamplingSeed 'GenerateRequest::sampling_seed
LunaTextRequestHandoffLease::sampling_seed
SamplingSeed::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti StopConditions 'GenerateRequest::stops
StopConditions::new
StopConditions::token_only'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TextInput TextInput::from_utf8
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TokenBuffer 'LunaTokenBufferLease::token_buffer
TokenBuffer::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TraceCorrelation 'GenerateRequest::trace
TraceCorrelation::from_ascii'
if ! rg -q --pcre2 -U \
    '^pub\(all\) enum Input \{\n  TokenIds\(TokenBuffer\)\n  Text\(TextInput\)\n\}$' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference Input authority variants drifted' >&2
  failed=1
fi
if ! rg -F -x -q \
    'pub fn Input::from_token_ids(ArrayView[Int], InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn Input::from_utf8(BytesView, InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn GenerateRequest::input(Self) -> Input' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn TokenBuffer::new(ArrayView[Int], InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn LunaTokenBufferLease::token_buffer(Self) -> TokenBuffer raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference input and token authority signatures drifted' >&2
  failed=1
fi
if rg -q --pcre2 -U \
    '^pub struct (CachePolicy|CacheScope|GenerateRequest|SamplingSeed|StopConditions|TextInput|TokenBuffer|TraceCorrelation) \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'payload-bearing inference request authority became debuggable' >&2
  failed=1
fi

for semantic_authority in \
    LunaRequestSemanticFailure \
    LunaRequestSemanticLease \
    LunaRequestSemanticProgress \
    LunaRequestSemanticState \
    LunaRequestSemanticStepBudget \
    LunaRequestSemanticStopTokenStatus \
    LunaRequestSemanticStorage \
    LunaRequestSemanticView \
    LunaRequestSemanticWork \
    LunaRequestSemanticWrite; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${semantic_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference semantic authority ${semantic_authority} is not opaque" >&2
    failed=1
  fi
done
if rg -q --pcre2 -U \
    '^pub struct LunaRequestSemantic[^ ]* \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority became debuggable' >&2
  failed=1
fi
if rg -n --pcre2 \
    '^pub fn LunaRequestSemantic[^:]*::.* -> .*\b(Array|ArrayView|ReadOnlyArray|FixedArray|Bytes|BytesView|String|StringView)\b' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority exposed a raw payload collection' >&2
  failed=1
fi
if rg -n '^pub fn LunaRequestSemantic[^:]*::.*epoch' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference semantic authority exposed a raw epoch' >&2
  failed=1
fi
if rg 'LunaRequestSemantic(Failure|Lease|Progress|State|StepBudget|StopTokenStatus|Storage|View|Work|Write)' \
    contracts/inference/pkg.generated.mbti | \
    rg -n -v '^pub (struct|fn) '; then
  printf '%s\n' \
    'inference semantic authority was embedded in another public type' >&2
  failed=1
fi
if rg -n --pcre2 \
    '^pub fn (?!LunaRequestSemantic[^:]*::)[^(]*\([^)]*LunaRequestSemantic(Failure|Lease|Progress|State|StepBudget|StopTokenStatus|Storage|View|Work|Write)[^)]*\)' \
    contracts/inference/pkg.generated.mbti | rg -F -v \
    'pub fn LunaTextRequestHandoffStorage::new(InferenceLimits, @spec.ModelIdentity, CacheScope, LunaRequestSemanticStepBudget) -> Self raise InferenceContractError'; then
  printf '%s\n' \
    'inference semantic authority gained a non-owner public consumer' >&2
  failed=1
fi

assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticFailure 'field
index
issue'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticLease 'prefix_view
release
stop_token_view
view'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticProgress 'is_failed
is_pending
is_ready'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticState 'is_failed
is_ready
is_validating'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStepBudget 'new
work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStopTokenStatus 'is_absent
is_present
is_stale
work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStorage 'begin
new
required_byte_cells
required_int_cells'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticView 'cache_permission
cache_scope_byte_at
cache_scope_length
inference_limits
is_stop_token
stop_string_byte_at
stop_string_count
stop_string_length
stop_token_at
stop_token_count'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWork 'abort
failure
last_work_units
progress
state
take_lease
total_work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWrite 'abort
finish
push_cache_scope_byte
push_stop_string_byte
push_stop_string_length
push_stop_token'

assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticFailure \
  LunaRequestSemanticWork::failure
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticLease \
  $'LunaRequestSemanticWork::take_lease\nLunaTextRequestHandoffClaim::take_semantics'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticProgress \
  $'LunaRequestSemanticWork::progress\nLunaTextRequestHandoffWork::progress'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticState \
  LunaRequestSemanticWork::state
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStepBudget \
  LunaRequestSemanticStepBudget::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStopTokenStatus \
  $'LunaRequestSemanticView::is_stop_token\nLunaRequestStopTokenRetentionSlot::is_stop_token\nLunaRequestStopTokenView::is_stop_token'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStorage \
  LunaRequestSemanticStorage::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticView \
  $'LunaRequestSemanticLease::view\nLunaTextRequestHandoffLease::semantic_view'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestPrefixView \
  LunaRequestSemanticLease::prefix_view
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWork \
  LunaRequestSemanticWrite::finish
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWrite \
  LunaRequestSemanticStorage::begin

for stop_authority in \
    LunaRequestPrefixView \
    LunaRequestStopTokenRetentionSlot \
    LunaRequestStopTokenView; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${stop_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference token-stop authority ${stop_authority} is not opaque" >&2
    failed=1
  fi
done
if rg -q --pcre2 -U \
    '^pub struct LunaRequest(?:StopToken[^ ]*|PrefixView) \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference token-stop authority became debuggable' >&2
  failed=1
fi
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestPrefixView 'binds_stop_tokens
cache_permission
is_live
scope_byte_at
scope_length'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenRetentionSlot 'is_live
is_stop_token
new
release'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenView 'is_live
is_stop_token
retain_into'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenRetentionSlot \
  LunaRequestStopTokenRetentionSlot::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenView \
  LunaRequestSemanticLease::stop_token_view

for model_identity_authority in \
  ContentDigest \
  LlamaModelMetadata \
  LlamaModelSpec \
  ModelIdentity \
  PlanDigest; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${model_identity_authority} \\{\\n  // private fields\\n\\}" \
      model/spec/pkg.generated.mbti; then
    printf '%s\n' \
      "model identity authority ${model_identity_authority} is not opaque" >&2
    failed=1
  fi
done
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti ContentDigest from_sha256
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti LlamaModelMetadata from_verified_content
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti LlamaModelSpec new
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti ModelIdentity new
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti PlanDigest from_sha256
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti ContentDigest 'ContentDigest::from_sha256
LlamaModelMetadata::content_digest
ModelIdentity::content'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti LlamaModelMetadata LlamaModelMetadata::from_verified_content
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti LlamaModelSpec 'LlamaModelMetadata::spec
LlamaModelSpec::new'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti ModelIdentity 'ModelIdentity::new'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti PlanDigest 'ModelIdentity::plan
PlanDigest::from_sha256'
if rg -n '^pub(\(all\))? type' \
    contracts/inference/pkg.generated.mbti model/spec/pkg.generated.mbti; then
  printf '%s\n' \
    'authority packages must not hide authority returns behind public aliases' >&2
  failed=1
fi
if ! rg -q '^pub fn ContentDigest::from_sha256\(String\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn PlanDigest::from_sha256\(String\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn ModelIdentity::new\(ContentDigest, PlanDigest\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelMetadata::content_digest\(Self\) -> ContentDigest$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelMetadata::from_verified_content\(LlamaModelSpec, ContentDigest\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelSpec::new\(vocabulary_size~ : Int, hidden_size~ : Int, intermediate_size~ : Int, layer_count~ : Int, attention_head_count~ : Int, kv_head_count~ : Int, context_length~ : Int, rope_theta~ : Double, rms_norm_epsilon\? : Double, weight_dtype~ : ModelDType, kv_dtype~ : ModelDType\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti; then
  printf '%s\n' 'model authority factory exclusivity drifted' >&2
  failed=1
fi

if [ -f contracts/inference/pkg.generated.mbti ] &&
  [ -f scheduler/core/pkg.generated.mbti ]; then
  if rg -n '^pub fn TokenBuffer::token_ids\(|^pub fn TokenizedRequest::(input_tokens|input_token_at)\(' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
    rg -n --pcre2 -U \
      'pub struct (TokenBuffer|TokenizedRequest) \{\n  (?!// private fields)' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
    rg -n --pcre2 -U \
      'pub struct (TokenBuffer|TokenizedRequest) \{(?s:[^}]*)\} derive\([^)]*Debug' \
      contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
    printf '%s\n' \
      'tokenized scheduler requests must remain opaque without raw token arrays' >&2
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
    printf '%s\n' \
      'scheduler token-only semantic authority surface drifted' >&2
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
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
