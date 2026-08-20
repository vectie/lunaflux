#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test scheduler/core \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/scheduler/core/core.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'scheduler cancellation release C output is missing' >&2
  exit 1
fi

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

extract_source_definition() {
  local pattern="$1"
  local source="$2"
  awk -v pattern="$pattern" '
    !copying && index($0, pattern) > 0 { copying = 1 }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (opens > 0) { entered = 1 }
      if (entered && depth == 0) exit
    }
  ' "$source"
}

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string'
contains_forbidden_allocation() {
  rg "$forbidden" | rg -v 'Error|Failure' | rg -q .
}

contains_strict_allocation() {
  rg "$forbidden" | rg -q .
}

contains_unexpected_admission_allocation() {
  rg "$forbidden" |
    rg -v \
      'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error63vectie_2flunaflux_2fscheduler_2fcore_2eSchedulerError_2eInvalid\)\)' |
    rg -q .
}

positive_body="$(extract_definition 'cancel__reservation__allocation__positive__control(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'scheduler cancellation allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'reserve__cancel__exact(' \
  'take__reserved__generated__token(' \
  'commit__cancel__reservation(' \
  'abort__cancel__reservation(' \
  'require__no__cancel__reservation(' \
  'has__exact__terminal(' \
  'clear__cancel__reservation(' \
  'allocate__request__generation(' \
  'enqueue__terminal(' \
  'remove__waiting(' \
  'recycle__slot(' \
  'release__active__request('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'scheduler cancellation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'scheduler cancellation path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done


for symbol in \
  'scheduler4core16TokenizedRequest29input__maximum__token__status(' \
  'contracts9inference11TokenBuffer22maximum__token__status(' \
  'contracts9inference11TokenBuffer8is__live(' \
  'contracts9inference22LunaTokenBufferStorage17is__sealed__epoch(' \
  'scheduler4core9Scheduler28commit__exclusive__admission('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'scheduler token-bound function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_strict_allocation; then
    printf 'scheduler token-bound function allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

preflight_body="$(extract_definition 'preflight__admission__request(')"
if [ -z "$preflight_body" ]; then
  printf '%s\n' 'scheduler admission preflight function is missing' >&2
  exit 1
fi
if printf '%s\n' "$preflight_body" |
  contains_unexpected_admission_allocation; then
  printf '%s\n' \
    'scheduler admission preflight contains a non-error allocation' >&2
  exit 1
fi
if printf '%s\n' "$preflight_body" | rg -q \
    'input__token__at|TokenBuffer[0-9]+token__status|while[[:space:]]*\(' ||
  [ "$(printf '%s\n' "$preflight_body" | rg -c \
    'input__maximum__token__status\(')" != '1' ]; then
  printf '%s\n' \
    'generated scheduler admission preflight rescans prompt token data' >&2
  exit 1
fi

admission_preflight="$(extract_source_definition \
  'fn Scheduler::preflight_admission_request(' \
  scheduler/core/admission.mbt)"
request_bound="$(extract_source_definition \
  'fn TokenizedRequest::input_maximum_token_status(' \
  scheduler/core/request.mbt)"
exclusive_commit="$(extract_source_definition \
  'pub fn Scheduler::commit_exclusive_admission(' \
  scheduler/core/admission.mbt)"
if [ -z "$admission_preflight" ] || [ -z "$request_bound" ] ||
  [ -z "$exclusive_commit" ] ||
  printf '%s\n' "$admission_preflight" | rg -q \
    'input_token_at|(^|[^A-Za-z_])token_status\(|(^|[[:space:]])(for|while|loop)([[:space:]]|$)' ||
  ! printf '%s\n' "$admission_preflight" | rg -q \
    'input_maximum_token_status' ||
  printf '%s\n' "$request_bound" | rg -q \
    'input_token_at|(^|[^A-Za-z_])token_status\(|(^|[[:space:]])(for|while|loop)([[:space:]]|$)' ||
  ! printf '%s\n' "$request_bound" | rg -q \
    'maximum_token_status' ||
  ! printf '%s\n' "$exclusive_commit" | rg -q --pcre2 -U \
    'input_maximum_token_status(?s:.*)!maximum_token\.is_within_bound\(\)(?s:.*)exclusive_admission_reserved = false(?s:.*)prepared\.consumed = true(?s:.*)ExclusiveAdmissionInvalidated(?s:.*)is_expired(?s:.*)install_admission_request'; then
  printf '%s\n' \
    'scheduler admission token-bound proof rescans prompt data or drifted' >&2
  exit 1
fi

scripts/validate-hot-path-allocations.sh
printf '%s\n' \
  'LunaFlux scheduler cancellation and admission allocation gate passed.'
