#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test \
  service/request_admission/pool_framed_wbtest.mbt \
  service/request_admission/direct_framed_preparation_hostile_wbtest.mbt \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/request_admission/request_admission.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'direct framed-preparation release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|double|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
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
  rg 'moonbit_malloc|moonbit_make_|Bytes4make|Array4new|moonbit_add_string' || true
}

# Error envelopes and constant-size authority/scalar shells are permitted.
# Payload arrays, Bytes, strings, maps, and variable-size owners remain visible.
contains_proportional_allocation() {
  allocation_lines |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error82vectie_2flunaflux_2fservice_2frequest__admission_2eRequestAdmissionError_2eInvalid\)' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error63vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2eInvalid\)' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC16option6OptionGRP(36vectie8lunaflux9tokenizer(17LunaTokenizerWork|23LunaTokenizerInputWrite)|46vectie8lunaflux9contracts9inference20LunaTokenBuffer(Write|Lease)|46vectie8lunaflux7service19incremental__output26LunaIncrementalOutputLease)E4Some\)' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC16option6OptionGdE4Some\)' |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0TP(46vectie8lunaflux7service18request__admission24LunaPreparedRequestClaim|46vectie8lunaflux9scheduler4core16TokenizedRequest)\)' |
    rg -q .
}

pool_new="$(extract_definition 'new__pool__with__clock(')"
if [ -z "$pool_new" ] ||
  ! printf '%s\n' "$pool_new" | allocation_lines | rg -q .; then
  printf '%s\n' \
    'direct framed-preparation allocation positive control is ineffective' >&2
  exit 1
fi

# This is the request-admission closure added by the direct framed route. The
# framed scanner itself has a separate exhaustive allocation gate.
for symbol in \
  'LunaRequestPreparationPool24try__begin__luna__framed(' \
  'LunaRequestPreparationPool19offer__framed__lane(' \
  'LunaRequestPreparationWork19offer__luna__framed(' \
  'LunaRequestPreparationPool8progress(' \
  'LunaRequestPreparationPool19progress__lane__one(' \
  'LunaRequestPreparationPool22abort__lane__resources(' \
  'LunaRequestPreparationPool10fail__lane(' \
  'LunaRequestPreparationPool25progress__framed__waiting(' \
  'LunaRequestPreparationPool25progress__framed__scanner(' \
  'LunaRequestPreparationPool28progress__framed__take__view(' \
  'LunaRequestPreparationPool29progress__framed__bind__model(' \
  'LunaRequestPreparationPool31progress__framed__bind__scalars(' \
  'LunaRequestPreparationPool30progress__framed__input__begin(' \
  'LunaRequestPreparationPool29progress__framed__input__copy(' \
  'LunaRequestPreparationPool33progress__framed__semantic__begin(' \
  'LunaRequestPreparationPool39progress__framed__semantic__token__copy(' \
  'LunaRequestPreparationPool42progress__framed__semantic__string__header(' \
  'LunaRequestPreparationPool40progress__framed__semantic__string__copy(' \
  'LunaRequestPreparationPool39progress__framed__semantic__cache__copy(' \
  'LunaRequestPreparationPool34progress__framed__semantic__finish(' \
  'LunaRequestPreparationPool31progress__framed__release__view(' \
  'LunaRequestPreparationPool23progress__output__setup(' \
  'LunaRequestPreparationPool18progress__assemble(' \
  'TokenizedRequest3new('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'direct framed-preparation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_proportional_allocation; then
    printf 'direct framed-preparation proportional path allocates: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

direct_sources=(
  service/request_admission/luna_framed_receipt.mbt
  service/request_admission/pool_framed_progress.mbt
  service/request_admission/pool_output_progress.mbt
)

if rg -n \
    'GenerateRequest::new|Input::(from_utf8|from_token_ids)|TextInput::|StopConditions::new|CachePolicy::new|materialize_luna|@utf8\.encode|Bytes::make|Map::|HashMap' \
    "${direct_sources[@]}"; then
  printf '%s\n' \
    'direct framed preparation reintroduced object materialization or growing storage' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'if length <= 0 \|\|[\s\S]*source_offset > source\.length\(\) - length \{[\s\S]*raise Invalid\(FrameContract\)[\s\S]*\}[\s\S]*let receipt = \(self\.clock_read\)\(\)' \
    service/request_admission/luna_framed_receipt.mbt ||
  ! rg -q --pcre2 -U \
    'if length == 0 \{[\s\S]*return 0[\s\S]*\}[\s\S]*let observed = \(self\.pool\.clock_read\)\(\)' \
    service/request_admission/luna_framed_receipt.mbt ||
  [ "$(rg -F -c 'let receipt = (self.clock_read)()' \
    service/request_admission/luna_framed_receipt.mbt)" != '1' ]; then
  printf '%s\n' \
    'direct framed receipt lost atomic range validation or single first-byte clock capture' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'if lane\.phase == LUNA_PREPARATION_FRAMED_RECEIVING[\s\S]*progress_framed_waiting\(lane\)[\s\S]*break[\s\S]*if lane\.total_work_units >= self\.work_limit\.work_units\(\)' \
    service/request_admission/pool_progress.mbt ||
  ! rg -q --pcre2 -U \
    'if charged != 1 \{[\s\S]*raise Invalid\(FrameContract\)[\s\S]*\}' \
    service/request_admission/pool_framed_progress.mbt ||
  ! rg -q --pcre2 -U \
    'lane\.total_work_units \+= charged\.to_uint64\(\)[\s\S]*used \+= charged' \
    service/request_admission/pool_progress.mbt; then
  printf '%s\n' \
    'direct framed FIFO receiving or exact scanner work accounting drifted' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'if index >= 128 \{[\s\S]*LUNA_PREPARATION_FRAMED_BIND_SCALARS[\s\S]*return[\s\S]*\}[\s\S]*content__digest__byte__at|if index >= 128 \{[\s\S]*LUNA_PREPARATION_FRAMED_BIND_SCALARS[\s\S]*return[\s\S]*\}[\s\S]*content_digest_byte_at' \
    service/request_admission/pool_framed_progress.mbt ||
  ! rg -q --pcre2 -U \
    'AdmissionDeadline::from_receipt\([\s\S]*lane\.receipt_at_millis[\s\S]*validate_pool_time\(lane\.receipt_at_millis, deadline, \(self\.clock_read\)\(\)\)[\s\S]*lane\.deadline = Some\(deadline\)' \
    service/request_admission/pool_framed_progress.mbt ||
  ! rg -q --pcre2 -U \
    'progress_framed_release_view[\s\S]*require_framed_view\(\)\.release\(\)[\s\S]*lane\.framed_view\.clear\(\)[\s\S]*LUNA_PREPARATION_(TEXT_TOKENIZE|SEMANTIC_VALIDATE)' \
    service/request_admission/pool_framed_progress.mbt ||
  ! rg -q --pcre2 -U \
    'lane\.claim = Some\(claim\)[\s\S]*validate_pool_time\([\s\S]*lane\.require_current_deadline\(\)[\s\S]*\(self\.clock_read\)\(\)[\s\S]*\)[\s\S]*lane\.phase = LUNA_PREPARATION_READY' \
    service/request_admission/pool_progress.mbt; then
  printf '%s\n' \
    'direct framed model/deadline/View-release ordering drifted' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'let framed_int_cells = @framed_wire\.LunaFramedRequestWorkspace::required_int_cells[\s\S]*checked_add_cells\(output_int_cells, framed_int_cells\)' \
    service/request_admission/pool_storage.mbt ||
  ! rg -q --pcre2 -U \
    'let framed_byte_cells = @framed_wire\.LunaFramedRequestWorkspace::required_byte_cells[\s\S]*checked_add_cells\([\s\S]*semantic_byte_cells,[\s\S]*framed_byte_cells' \
    service/request_admission/pool_storage.mbt ||
  ! rg -q --pcre2 -U \
    'checked_add_cells\(6UL, output_reference_cells\)[\s\S]*lanes' \
    service/request_admission/pool_storage.mbt; then
  printf '%s\n' \
    'direct framed scanner storage is absent from checked pool aggregates' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux direct framed-preparation allocation gate passed.'
