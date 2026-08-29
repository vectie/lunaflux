#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ {
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
  ' "$generated_c"
}

allocation_lines() {
  rg 'moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string' || true
}

# Only constant authority wrappers are permitted after pool startup. Error
# envelopes are excluded exactly; payload arrays, Bytes, strings, maps, and
# variable-size owners remain visible. The allowed Some wrappers merely store
# already-preallocated generation-authenticated valtypes.
contains_unbounded_allocation() {
  allocation_lines |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC16option6OptionGRP(36vectie8lunaflux9tokenizer(17LunaTokenizerWork|23LunaTokenizerInputWrite)|46vectie8lunaflux9contracts9inference(20LunaTokenBuffer(Write|Lease)|21LunaRequestPrefixView)|46vectie8lunaflux7service19incremental__output2[56]LunaIncrementalOutput(Work|Lease))E4Some\)' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC16option6OptionGdE4Some\)' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0TP(46vectie8lunaflux7service18request__admission(19LunaPreparedRequest|24LunaPreparedRequestClaim)|46vectie8lunaflux9scheduler4core16TokenizedRequest)\)' |
    rg -q .
}

pool_new="$(extract_definition 'LunaRequestPreparationPool3new(')"
if [ -z "$pool_new" ] ||
  ! printf '%s\n' "$pool_new" | allocation_lines | rg -q .; then
  printf '%s\n' 'preparation-pool allocation positive control is ineffective' >&2
  exit 1
fi

assemble="$(extract_definition 'LunaRequestPreparationPool18progress__assemble(')"
take_prepared="$(extract_definition 'LunaRequestPreparationWork14take__prepared(')"
tokenized_new="$(extract_definition 'new__with__prefix(')"
if [ -z "$assemble" ] || [ -z "$take_prepared" ] ||
  [ -z "$tokenized_new" ] ||
  ! printf '%s\n' "$assemble" |
    rg -q 'moonbit_malloc\(sizeof\(struct .*LunaPreparedRequestClaim\)' ||
  ! printf '%s\n' "$take_prepared" |
    rg -q 'moonbit_malloc\(sizeof\(struct .*LunaPreparedRequest\)' ||
  ! printf '%s\n' "$tokenized_new" |
    rg -q 'moonbit_malloc\(sizeof\(struct .*TokenizedRequest\)'; then
  printf '%s\n' \
    'constant prepared/claim/request shell allocation controls are missing' >&2
  exit 1
fi

# This is the transitive proportional-work closure used by central progress.
# Every symbol must remain emitted so inlining cannot silently narrow evidence.
for symbol in \
  'LunaRequestPreparationPool11try__submit(' \
  'LunaRequestPreparationPool46try__begin__luna__text__handoff__with__receipt(' \
  'LunaRequestPreparationPool11start__lane(' \
  'LunaRequestPreparationPool8progress(' \
  'LunaRequestPreparationPool19progress__lane__one(' \
  'LunaRequestPreparationPool22abort__lane__resources(' \
  'LunaRequestPreparationPool10fail__lane(' \
  'LunaRequestPreparationPool27progress__text__input__copy(' \
  'LunaRequestPreparationPool24progress__text__tokenize(' \
  'LunaRequestPreparationPool27progress__text__begin__copy(' \
  'LunaRequestPreparationPool20progress__text__copy(' \
  'LunaRequestPreparationPool29progress__typed__input__begin(' \
  'LunaRequestPreparationPool28progress__typed__input__copy(' \
  'LunaRequestPreparationPool34progress__semantic__cache__measure(' \
  'LunaRequestPreparationPool25progress__semantic__begin(' \
  'LunaRequestPreparationPool31progress__semantic__token__copy(' \
  'LunaRequestPreparationPool35progress__semantic__string__measure(' \
  'LunaRequestPreparationPool34progress__semantic__string__header(' \
  'LunaRequestPreparationPool32progress__semantic__string__emit(' \
  'LunaRequestPreparationPool31progress__semantic__cache__emit(' \
  'LunaRequestPreparationPool26progress__semantic__finish(' \
  'LunaRequestPreparationPool28progress__semantic__validate(' \
  'LunaRequestPreparationPool24progress__semantic__take(' \
  'luna__semantic__codepoint__at(' \
  'luna__semantic__codepoint__units__at(' \
  'luna__semantic__utf8__width(' \
  'luna__semantic__utf8__byte(' \
  'LunaRequestPreparationPool23progress__output__setup(' \
  'LunaRequestPreparationPool18progress__assemble(' \
  'LunaRequestPreparationWork5state(' \
  'LunaRequestPreparationWork14take__prepared(' \
  'LunaPreparedRequestClaim7release(' \
  'LunaTokenizerWorker18begin__luna__input(' \
  'LunaTokenizerInputWrite15require__worker(' \
  'LunaTokenizerInputWrite10push__byte(' \
  'LunaTokenizerInputWrite6finish(' \
  'LunaTokenizerInputWrite5abort(' \
  'LunaTokenizerWork8progress(' \
  'LunaTokenizerWork5abort(' \
  'LunaTokenizerWork16copy__tokens__to(' \
  'LunaTokenBufferStorage5begin(' \
  'LunaTokenBufferWrite4push(' \
  'LunaTokenBufferWrite4seal(' \
  'LunaRequestSemanticStorage5begin(' \
  'LunaRequestSemanticWrite17push__stop__token(' \
  'LunaRequestSemanticWrite26push__stop__string__length(' \
  'LunaRequestSemanticWrite24push__stop__string__byte(' \
  'LunaRequestSemanticWrite24push__cache__scope__byte(' \
  'LunaRequestSemanticWrite6finish(' \
  'LunaRequestSemanticWork8progress(' \
  'LunaRequestSemanticWork11take__lease(' \
  'LunaTextRequestHandoffLease11take__claim(' \
  'LunaTextRequestHandoffLease14prompt__length(' \
  'LunaTextRequestHandoffClaim15take__semantics(' \
  'LunaTextRequestHandoffClaim16prompt__byte__at(' \
  'LunaTextRequestHandoffClaim7release(' \
  'LunaIncrementalOutputWorkspace31begin__luna__request__semantics(' \
  'LunaIncrementalOutputWork8progress(' \
  'LunaIncrementalOutputWork5abort(' \
  'LunaIncrementalOutputWork11take__lease(' \
  'new__with__prefix('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'preparation-pool allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_unbounded_allocation; then
    printf 'preparation-pool proportional path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

# These immutable request projections can be inlined away in release C. Pin
# their exact scalar/field form so the one-byte caller above cannot hide a
# proportional conversion or fresh Bytes construction behind an out-of-line
# accessor.
if ! rg -q --pcre2 -U \
    'pub fn GenerateRequest::input\(self : GenerateRequest\) -> Input \{\n  self\.input\n\}' \
    contracts/inference/request.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn TextInput::utf8_bytes\(self : TextInput\) -> Int \{\n  self\.utf8\.length\(\)\n\}' \
    contracts/inference/input.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn TextInput::utf8\(self : TextInput\) -> Bytes \{\n  self\.utf8\n\}' \
    contracts/inference/input.mbt; then
  printf '%s\n' \
    'preparation-pool text source projections are no longer scalar retained-data reads' >&2
  exit 1
fi

if rg -n 'begin_bytes' service/request_admission/pool*.mbt ||
  ! rg -q --pcre2 -U \
    'progress_text_input_copy(?s).*if lane\.copy_index < text\.utf8_bytes\(\)(?s).*write\.push_byte\(text\.utf8\(\)\[lane\.copy_index\]\)(?s).*lane\.copy_index \+= 1(?s).*return(?s).*write\.finish\(\)' \
    service/request_admission/pool_progress.mbt ||
  ! rg -q --pcre2 -U \
    'SpecialTokenRejected\(token_id=_\) \|(?s).*OutputTooLong\(actual=_, maximum=_\)(?s).*lane\.tokenizer_work = None(?s).*raise Invalid\(Tokenization\)' \
    service/request_admission/pool_progress.mbt; then
  printf '%s\n' \
    'preparation-pool text ingress is not exact one-byte work with terminal cleanup' >&2
  exit 1
fi

lane_constructor_source="service/request_admission/pool_lane_new.mbt"
if rg -n '(^|[^A-Za-z])Array::|Map::|HashMap|@utf8\.encode|Bytes::make' \
    service/request_admission/pool_progress.mbt \
    service/request_admission/pool_semantic_progress.mbt \
    service/request_admission/pool_work.mbt ||
  rg -n '(^|[^A-Za-z])Array::|Map::|HashMap|@utf8\.encode|Bytes::make' \
    service/request_admission/pool.mbt ||
  [ "$(rg -c 'Array::new\(capacity=1\)' "$lane_constructor_source")" != '6' ] ||
  ! rg -q '^fn new_preparation_lane\(' "$lane_constructor_source" ||
  rg -n 'Map::|HashMap|@utf8\.encode|Bytes::make' "$lane_constructor_source" ||
  rg -n '(^|[^A-Za-z])Array::' "$lane_constructor_source" |
    rg -v 'Array::new\(capacity=1\)'; then
  printf '%s\n' \
    'preparation-pool progress introduced a growing/proportional collection' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux request-preparation pool allocation gate passed.'
