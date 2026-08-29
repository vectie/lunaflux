#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

moon test service/openai_compat --target native --release --deny-warn --warn-list +73

generated_c_files=(
  "_build/native/release/test/service/openai_compat/openai_compat.whitebox_test.c"
  "_build/native/release/test/service/openai_compat/openai_compat.blackbox_test.c"
)
for generated_c in "${generated_c_files[@]}"; do
  if [ ! -f "$generated_c" ]; then
    printf 'Luna OpenAI compatibility release C is missing: %s\n' \
      "$generated_c" >&2
    exit 1
  fi
done

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /_M0/ && $0 ~ /\($/ {
      candidate = 1; body = $0 ORS; next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) {
        copying = 1; depth = 1; printf "%s", body; candidate = 0; next
      }
    }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "${generated_c_files[@]}"
}

forbidden='moonbit_malloc|moonbit_make_|moonbit_add_string|memcpy|memmove'
contains_success_allocation_or_copy() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc.*LunaOpenAICompatError_2eLunaOpenAICompatInvalid' |
    rg -v 'moonbit_malloc.*LunaEventError_2eLunaEventInvalid' |
    rg -v 'moonbit_malloc.*InferenceContractError_2eInvalid' |
    rg -q .
}

positive="$(extract_definition 'LunaOpenAICompatWorkspace3new(')"
if [ -z "$positive" ] ||
  ! printf '%s\n' "$positive" | rg -q "$forbidden"; then
  printf '%s\n' 'Luna OpenAI compatibility allocation positive control failed' >&2
  exit 1
fi

for symbol in \
  'LunaOpenAICompatWorkspace5begin(' \
  'LunaOpenAICompatWork8progress(' \
  'LunaOpenAICompatWorkspace13progress__one(' \
  'LunaOpenAICompatWorkspace19authenticate__event(' \
  'LunaOpenAICompatWorkspace15progress__setup(' \
  'LunaOpenAICompatWorkspace15setup__accepted(' \
  'LunaOpenAICompatWorkspace12setup__token(' \
  'LunaOpenAICompatWorkspace12setup__usage(' \
  'LunaOpenAICompatWorkspace16setup__completed(' \
  'LunaOpenAICompatWorkspace13setup__failed(' \
  'LunaOpenAICompatWorkspace15configure__part(' \
  'LunaOpenAICompatWorkspace17progress__literal(' \
  'LunaOpenAICompatWorkspace13progress__raw(' \
  'LunaOpenAICompatWorkspace22progress__load__escape(' \
  'LunaOpenAICompatWorkspace17payload__byte__at(' \
  'LunaOpenAICompatWorkspace22progress__emit__escape(' \
  'LunaOpenAICompatWorkspace26progress__decimal__prepare(' \
  'LunaOpenAICompatWorkspace23progress__decimal__emit(' \
  'LunaOpenAICompatWorkspace10emit__byte(' \
  'luna__openai__escape__kind(' \
  'luna__openai__escaped__byte(' \
  'luna__openai__literal(' \
  'luna__openai__part__kind(' \
  'LunaOpenAICompatWork10take__view(' \
  'LunaOpenAICompatView15copy__chunk__to(' \
  'LunaOpenAICompatView7release(' \
  'LunaOpenAIInboundWorkspace5begin(' \
  'LunaOpenAIInboundWork8progress(' \
  'LunaOpenAIInboundWorkspace22progress__inbound__one(' \
  'LunaOpenAIInboundWorkspace18authenticate__http(' \
  'LunaOpenAIInboundWorkspace15progress__parse(' \
  'LunaOpenAIInboundWorkspace13start__number(' \
  'LunaOpenAIInboundWorkspace23add__significand__digit(' \
  'LunaOpenAIInboundWorkspace20finish__number__scan(' \
  'LunaOpenAIInboundWorkspace16progress__number(' \
  'LunaOpenAIInboundWorkspace23progress__number__scale(' \
  'LunaOpenAIInboundWorkspace23progress__number__final(' \
  'LunaOpenAIInboundWorkspace13begin__render(' \
  'LunaOpenAIInboundWorkspace14progress__true(' \
  'LunaOpenAIInboundWorkspace16progress__string(' \
  'LunaOpenAIInboundWorkspace22progress__string__emit(' \
  'LunaOpenAIInboundWorkspace20accept__string__byte(' \
  'LunaOpenAIInboundWorkspace18advance__raw__utf8(' \
  'LunaOpenAIInboundWorkspace14finish__string(' \
  'LunaOpenAIInboundWorkspace15finish__message(' \
  'LunaOpenAIInboundWorkspace11finish__top(' \
  'LunaOpenAIInboundWorkspace25progress__render__segment(' \
  'LunaOpenAIInboundWorkspace24progress__render__string(' \
  'LunaOpenAIInboundWorkspace22progress__render__emit(' \
  'LunaOpenAIInboundWorkspace18emit__prompt__byte(' \
  'LunaOpenAIInboundWorkspace14begin__handoff(' \
  'LunaOpenAIInboundWorkspace23progress__handoff__push(' \
  'LunaOpenAIInboundWorkspace17progress__handoff(' \
  'luna__json__hex(' \
  'luna__json__space(' \
  'luna__known__byte(' \
  'luna__known__length(' \
  'luna__json__codepoint__byte(' \
  'luna__json__codepoint__width(' \
  'LunaOpenAIInboundWork10take__view(' \
  'LunaOpenAIInboundView15copy__chunk__to(' \
  'LunaOpenAIInboundView7release(' \
  'LunaValidatedSampling6sample(' \
  'LunaTextRequestHandoffStorage5begin(' \
  'LunaTextRequestHandoffWrite18push__prompt__byte(' \
  'LunaTextRequestHandoffWrite17push__stop__token(' \
  'LunaTextRequestHandoffWrite26push__stop__string__length(' \
  'LunaTextRequestHandoffWrite24push__stop__string__byte(' \
  'LunaTextRequestHandoffWrite6finish(' \
  'LunaTextRequestHandoffWork8progress(' \
  'LunaTextRequestHandoffWork11take__lease(' \
  'LunaTextRequestHandoffLease14semantic__view(' \
  'LunaTextRequestHandoffLease7release('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna OpenAI compatibility function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_success_allocation_or_copy; then
    printf 'Luna OpenAI warmed path allocates or copies proportionally: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

mbti="service/openai_compat/pkg.generated.mbti"
for type in \
  LunaOpenAICompatStepBudget \
  LunaOpenAICompatWorkspace \
  LunaOpenAICompatWork \
  LunaOpenAICompatView \
  LunaOpenAIChatTemplate \
  LunaOpenAIInboundLimits \
  LunaOpenAIInboundWorkspace \
  LunaOpenAIInboundWork \
  LunaOpenAIInboundView; do
  if ! rg -U -q "pub struct ${type} \{\n  // private fields\n\}" "$mbti"; then
    printf 'Luna OpenAI compatibility type is not opaque: %s\n' "$type" >&2
    exit 1
  fi
  if rg -q "impl Debug for ${type}" "$mbti"; then
    printf 'Luna OpenAI compatibility authority is Debug: %s\n' "$type" >&2
    exit 1
  fi
done
if rg -q -- 'LunaOpenAI(Compat|Inbound|ChatTemplate).*(-> (FixedArray|Array|ReadOnlyArray|ArrayView|Bytes|String|@luna_event|@http1)|epoch|storage|semantic_views|http_views|frame_views)' "$mbti"; then
  printf '%s\n' 'Luna OpenAI compatibility authority surface leaked' >&2
  exit 1
fi
if [ "$(rg -c '^pub fn LunaOpenAICompatStepBudget::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaOpenAICompatWorkspace::' "$mbti")" -ne 4 ] ||
  [ "$(rg -c '^pub fn LunaOpenAICompatWork::' "$mbti")" -ne 7 ] ||
  [ "$(rg -c '^pub fn LunaOpenAICompatView::' "$mbti")" -ne 3 ] ||
  [ "$(rg -c '^pub fn LunaOpenAIChatTemplate::' "$mbti")" -ne 3 ] ||
  [ "$(rg -c '^pub fn LunaOpenAIInboundLimits::' "$mbti")" -ne 1 ] ||
  [ "$(rg -c '^pub fn LunaOpenAIInboundWorkspace::' "$mbti")" -ne 5 ] ||
  [ "$(rg -c '^pub fn LunaOpenAIInboundWork::' "$mbti")" -ne 7 ] ||
  [ "$(rg -c '^pub fn LunaOpenAIInboundView::' "$mbti")" -ne 5 ]; then
  printf '%s\n' 'Luna OpenAI compatibility method surface drifted' >&2
  exit 1
fi
if rg -n 'StringBuilder|Json::|JSON::|JsonValue|JsonObject|Bytes::make|@utf8\.encode|GenerateRequest|TextInput|copy_delta_to|copy_final_delta_to|copy_code_to' \
    service/openai_compat --glob='*.mbt' |
  rg -v '_test\.mbt|_wbtest\.mbt'; then
  printf '%s\n' 'Luna OpenAI compatibility direct-encoding source drifted' >&2
  exit 1
fi
if [ "$(rg -c 'semantic_views: Array::new\(capacity=1\)' service/openai_compat/workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'http_views: Array::new\(capacity=1\)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'request_ids: Array::new\(capacity=1\)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'handoff_writes: Array::new\(capacity=1\)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'handoff_works: Array::new\(capacity=1\)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'handoff_leases: Array::new\(capacity=1\)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'semantic reference-slot invariant' service/openai_compat/workspace.mbt)" -ne 1 ] ||
  [ "$(rg -c 'inbound reference-slot invariant' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  rg -q 'Option\[@luna_event\.LunaEventView\]' service/openai_compat; then
  printf '%s\n' 'Luna OpenAI semantic reference storage drifted' >&2
  exit 1
fi
if rg -n '@framed_wire|LunaFramedTextRequest|RequestFrameBuffer' \
    service/openai_compat --glob='*.mbt' | rg -v '_test\.mbt|_wbtest\.mbt'; then
  printf '%s\n' 'Luna OpenAI production inbound regained framed-wire coupling' >&2
  exit 1
fi
if [ "$(rg -F -c 'stop_tokens: FixedArray::make(inference.max_stop_token_ids(), 0)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -F -c 'stop_string_cells: FixedArray::make(inference.max_stop_strings() * 2, 0)' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  [ "$(rg -F -c 'stop_string_bytes: FixedArray::make(' service/openai_compat/inbound_workspace.mbt)" -ne 1 ] ||
  rg -q '(^|[^A-Za-z])Array\[(Int|Byte|String)\]|Bytes|StringBuilder|Json::' \
    service/openai_compat/inbound_{types,workspace,parser,string,number,stops,lifecycle}.mbt; then
  printf '%s\n' 'Luna OpenAI sampling storage or parser allocation shape drifted' >&2
  exit 1
fi
if [ "$(rg -F -c '@sampling.stochastic_sample_at_scalars(' engine/device_step/paged_executor_wire_completion.mbt)" -ne 1 ] ||
  ! rg -q 'request.sampling_seed()' service/openai_compat/inbound_sampling_wbtest.mbt ||
  ! rg -F -q '@sampling.stochastic_sample_at_scalars(' service/openai_compat/inbound_sampling_wbtest.mbt; then
  printf '%s\n' 'Luna OpenAI canonical-to-device sampling evidence drifted' >&2
  exit 1
fi

printf '%s\n' 'Luna OpenAI compatibility allocation and API gate passed.'
