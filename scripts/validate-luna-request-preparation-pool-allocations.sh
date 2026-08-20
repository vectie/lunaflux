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
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC16option6OptionGRP(36vectie8lunaflux9tokenizer17LunaTokenizerWork|46vectie8lunaflux9contracts9inference20LunaTokenBuffer(Write|Lease)|46vectie8lunaflux7service19incremental__output2[56]LunaIncrementalOutput(Work|Lease))E4Some\)' |
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
tokenized_new="$(extract_definition 'TokenizedRequest3new(')"
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
  'LunaRequestPreparationPool11start__lane(' \
  'LunaRequestPreparationPool8progress(' \
  'LunaRequestPreparationPool19progress__lane__one(' \
  'LunaRequestPreparationPool24progress__text__tokenize(' \
  'LunaRequestPreparationPool27progress__text__begin__copy(' \
  'LunaRequestPreparationPool20progress__text__copy(' \
  'LunaRequestPreparationPool23progress__output__setup(' \
  'LunaRequestPreparationPool18progress__assemble(' \
  'LunaRequestPreparationWork5state(' \
  'LunaRequestPreparationWork14take__prepared(' \
  'LunaPreparedRequestClaim7release(' \
  'LunaTokenizerWorker12begin__bytes(' \
  'LunaTokenizerWork8progress(' \
  'LunaTokenizerWork16copy__tokens__to(' \
  'LunaTokenBufferStorage5begin(' \
  'LunaTokenBufferWrite4push(' \
  'LunaTokenBufferWrite4seal(' \
  'LunaIncrementalOutputWorkspace5begin(' \
  'LunaIncrementalOutputWork8progress(' \
  'LunaIncrementalOutputWork11take__lease(' \
  'TokenizedRequest3new('; do
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

if rg -n '(^|[^A-Za-z])Array::|Map::|HashMap|@utf8\.encode|Bytes::make' \
    service/request_admission/pool*.mbt; then
  printf '%s\n' \
    'preparation-pool progress introduced a growing/proportional collection' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux request-preparation pool allocation gate passed.'
