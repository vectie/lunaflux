#!/usr/bin/env bash
set -eu

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

package='benchmarks/openai_adapter'

if rg -n \
  'moonbitlang/async|async/socket|async/fs|@socket|@fs|@env|@process|@cuda|@hardware|Python|PyTorch|vllm/|sglang/' \
  "$package" --glob '*.mbt' --glob 'moon.pkg'; then
  fail 'OpenAI baseline adapter gained process network filesystem or device authority'
fi

for required in \
  'openai_comparison_engines' \
  'require_identical_openai_comparison_inputs' \
  'left.tokenizer != right.tokenizer' \
  'left.corpus != right.corpus' \
  'left.protocol != right.protocol' \
  'left.workload != right.workload' \
  'openai_baseline_token_counts_equal' \
  '@evidence.maximum_request_events_per_summary()' \
  'maximum_event_bytes > 1048576' \
  'maximum_json_depth > 32'; do
  if ! rg -Fq "$required" "$package"; then
    fail "baseline identity/input/storage boundary lost: $required"
  fi
done

for required in \
  'collector.submit' \
  'collector.admit' \
  'collector.first_token' \
  'collector.terminate' \
  'collector.finish' \
  'outcome=Completed' \
  'generated_tokens=lane.completion_tokens'; do
  if ! rg -Fq "$required" "$package"; then
    fail "baseline collector lifecycle boundary lost: $required"
  fi
done

for required in \
  'choices.length() > 1' \
  'name != "role" && name != "content"' \
  'reason == "stop" || reason == "length"' \
  '@json_guard.reject_duplicate_object_keys' \
  '!lane.finish_seen || !lane.usage_seen' \
  'completion != lane.observed_tokens' \
  'openai_baseline_choice_token_count' \
  'lane.event_length >= lane.event_storage.length()' \
  'OpenAIBaselineAmbiguousEvent'; do
  if ! rg -Fq "$required" "$package"; then
    fail "baseline ambiguous-SSE rejection lost: $required"
  fi
done

for required in \
  'BenchmarkEngineIdentity::new' \
  'BenchmarkWorkload::new' \
  'OpenAIComparisonMeasurementSet' \
  'admit_openai_comparison_measurements' \
  'observation_protocol=openai.chat.completions.sse.v1' \
  'declared_request_protocol_sha256=' \
  'protocol=trials[0].observed_protocol_digest' \
  'schema=lunaflux-openai-baseline-container-identity.v1' \
  'comparison_executable_sha256=' \
  'schema=lunaflux-openai-baseline-trial.v1' \
  'process_lifecycle_authority=external' \
  'credential_authority=external'; do
  if ! rg -Fq "$required" "$package"; then
    fail "baseline canonical comparison bridge lost: $required"
  fi
done

for required in \
  'OpenAIBaselineProtocolIncompatible' \
  'require_responses_v1_observation' \
  'raise OpenAIBaselineProtocolIncompatible' \
  'caller digest cannot relabel Chat SSE evidence as Responses' \
  'Chat measurement set refuses Responses approval'; do
  if ! rg -Fq "$required" "$package"; then
    fail "Chat/Responses incompatibility boundary lost: $required"
  fi
done

responses_public_api=$(rg -n '^pub(\(all\))? (struct|enum|suberror|fn)' \
  "$package" --glob '*.mbt' --glob '!**/*test.mbt' || true)
responses_public_symbols=$(
  printf '%s\n' "$responses_public_api" |
    sed -E -n \
      -e 's#^.*:pub(\(all\))? (struct|enum|suberror) (OpenAIResponses[A-Za-z0-9_]+).*#type \2 \3#p' \
      -e 's#^.*:pub fn ((OpenAIResponses[^(:]*::[^ (]+|admit_openai_responses_comparison_measurements))\(.*#fn \1#p' |
    LC_ALL=C sort
)
expected_responses_public_symbols=$(
  cat <<'EOF' | LC_ALL=C sort
fn OpenAIResponsesComparisonMeasurementSet::engine_identity
fn OpenAIResponsesComparisonMeasurementSet::environment
fn OpenAIResponsesComparisonMeasurementSet::measurements
fn OpenAIResponsesComparisonMeasurementSet::require_responses_v1_observation
fn OpenAIResponsesComparisonMeasurementSet::trial_count
fn OpenAIResponsesComparisonMeasurementSet::workload
fn OpenAIResponsesTrialContract::new
fn OpenAIResponsesTrialContract::start
fn OpenAIResponsesTrialContract::start_captured
fn OpenAIResponsesTrialEvidence::canonical_bytes
fn OpenAIResponsesTrialEvidence::comparison_executable_digest
fn OpenAIResponsesTrialEvidence::configuration_digest
fn OpenAIResponsesTrialEvidence::corpus_digest
fn OpenAIResponsesTrialEvidence::declared_request_protocol_digest
fn OpenAIResponsesTrialEvidence::digest
fn OpenAIResponsesTrialEvidence::engine_identity
fn OpenAIResponsesTrialEvidence::image_digest
fn OpenAIResponsesTrialEvidence::measurements
fn OpenAIResponsesTrialEvidence::observed_protocol
fn OpenAIResponsesTrialEvidence::protocol_digest
fn OpenAIResponsesTrialEvidence::revision_digest
fn OpenAIResponsesTrialEvidence::tokenizer_digest
fn OpenAIResponsesTrialEvidence::workload
fn OpenAIResponsesTrialObserver::finish
fn OpenAIResponsesTrialObserver::observe_response_head
fn OpenAIResponsesTrialObserver::observe_response_head_captured
fn OpenAIResponsesTrialObserver::offer_sse_bytes
fn OpenAIResponsesTrialObserver::offer_sse_bytes_captured
fn OpenAIResponsesTrialObserver::submit
fn OpenAIResponsesTrialObserver::submit_captured
fn OpenAIResponsesTrialObserver::terminate_transport
fn OpenAIResponsesTrialObserver::terminate_transport_captured
fn admit_openai_responses_comparison_measurements
type struct OpenAIResponsesComparisonMeasurementSet
type struct OpenAIResponsesTrialContract
type struct OpenAIResponsesTrialEvidence
type struct OpenAIResponsesTrialObserver
EOF
)
if [ "$responses_public_symbols" != "$expected_responses_public_symbols" ]; then
  printf '%s\n%s\n' \
    'OpenAI Responses adapter exposes an unreviewed public API:' \
    "$responses_public_symbols" >&2
  failed=1
fi

for required in \
  'let observed_protocol = ResponsesSseV1' \
  'observation_protocol=openai.responses.sse.v1' \
  'admit_openai_responses_comparison_measurements' \
  'OpenAIResponsesTrialContract::start_captured' \
  'OpenAIResponsesTrialObserver::submit_captured' \
  'OpenAIResponsesTrialObserver::observe_response_head_captured' \
  'OpenAIResponsesTrialObserver::offer_sse_bytes_captured' \
  'OpenAIResponsesTrialObserver::terminate_transport_captured' \
  '@runner.captured_monotonic_nanos_measurement_digest().as_hex()' \
  'captured_mode: false' \
  'captured_mode: true' \
  'if self.captured_mode {' \
  'if !self.captured_mode {' \
  'comparison_authority=none' \
  'correctness_authority=none' \
  'Responses observer rejects reordered and inconsistent lifecycle events' \
  'Responses partial transport failure without usage fails closed' \
  'shared Responses lifecycle accepts terminal-only usage and auxiliary events' \
  'Responses measurement bridge admits exact inert 81-trial matrix' \
  'Responses measurement bridge rejects one image-pin substitution'; do
  if ! rg -Fq "$required" "$package"; then
    fail "Responses/captured observation boundary lost: $required"
  fi
done

for opaque_type in \
  OpenAIResponsesTrialContract \
  OpenAIResponsesTrialObserver \
  OpenAIResponsesTrialEvidence \
  OpenAIResponsesComparisonMeasurementSet; do
  if ! rg -U -q \
    "pub struct ${opaque_type} \\{\\n  // private fields\\n\\}" \
    "$package/pkg.generated.mbti"; then
    fail "Responses generated interface is not opaque: ${opaque_type}"
  fi
done

if rg -n \
  'correctness_passed=true|BenchmarkTrial::from_summary|pub fn OpenAIBaselineTrialEvidence::trial' \
  "$package" --glob '*.mbt'; then
  fail 'OpenAI observation adapter gained correctness-passed authority'
fi

for hostile in \
  'fake baseline server rejects ambiguous SSE semantics' \
  'DONE requires one finish and exact usage' \
  'response head and transport outcome mappings are closed' \
  'bounded SSE storage rejects an oversized fake-server event' \
  'content requires exact token observations even on transport failure' \
  'vLLM and SGLang declarations require identical shared inputs' \
  'all three engines share the exact split Chat SSE lifecycle' \
  'branded OpenAI measurement set has exact three by nine by three shape' \
  'comparison measurement set rejects one protocol substitution'; do
  if ! rg -Fq "$hostile" "$package"; then
    fail "baseline hostile fake-server fixture lost: $hostile"
  fi
done

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "OpenAI baseline source exceeds file budget: $file ($lines)"
  fi
done < <(find "$package" -name '*.mbt' -type f | sort)

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'OpenAI baseline adapter boundary gate passed'
