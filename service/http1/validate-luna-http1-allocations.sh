#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

moon test service/http1 --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/http1/http1.whitebox_test.c"
mbti="service/http1/pkg.generated.mbti"
for required in "$generated_c" "$mbti"; do
  if [ ! -f "$required" ]; then
    printf 'Luna HTTP/1 evidence is missing: %s\n' "$required" >&2
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
  ' "$generated_c"
}

forbidden='moonbit_malloc|moonbit_make_|moonbit_add_string|memcpy|memmove'
contains_success_allocation_or_copy() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc.*LunaHttp1Error_2eLunaHttp1Failed' |
    rg -v 'moonbit_malloc.*LunaApiAuthError_2eLunaApiAuthFailed' |
    rg -q .
}

positive="$(extract_definition 'LunaHttp1Workspace3new(')"
if [ -z "$positive" ] ||
  ! printf '%s\n' "$positive" | rg -q 'moonbit_make_|moonbit_malloc'; then
  printf '%s\n' 'Luna HTTP/1 allocation positive control failed' >&2
  exit 1
fi

response_positive="$(extract_definition 'LunaHttp1ResponseWorkspace3new(')"
if [ -z "$response_positive" ] ||
  ! printf '%s\n' "$response_positive" | rg -q 'moonbit_make_|moonbit_malloc'; then
  printf '%s\n' 'Luna HTTP/1 response allocation positive control failed' >&2
  exit 1
fi

for symbol in \
  'LunaHttp1Workspace5begin(' \
  'LunaHttp1Workspace21close__authentication(' \
  'LunaHttp1Workspace16reset__operation(' \
  'LunaHttp1Workspace11clear__auth(' \
  'LunaHttp1Workspace4fail(' \
  'LunaHttp1Workspace14failure__value(' \
  'LunaHttp1Workspace6charge(' \
  'LunaHttp1Work18require__workspace(' \
  'LunaHttp1Work5offer(' \
  'LunaHttp1Workspace18accept__head__byte(' \
  'LunaHttp1Workspace18accept__body__byte(' \
  'LunaHttp1Work13finish__input(' \
  'LunaHttp1Work5state(' \
  'LunaHttp1Work7failure(' \
  'LunaHttp1Work10take__view(' \
  'LunaHttp1Work5abort(' \
  'LunaHttp1Work8progress(' \
  'LunaHttp1Workspace13progress__one(' \
  'LunaHttp1Workspace25progress__scan__line__one(' \
  'LunaHttp1Workspace22progress__request__one(' \
  'LunaHttp1Workspace27progress__header__name__one(' \
  'LunaHttp1Workspace28progress__header__value__one(' \
  'LunaHttp1Workspace21finish__header__value(' \
  'LunaHttp1Workspace23progress__required__one(' \
  'LunaHttp1Workspace26progress__auth__begin__one(' \
  'LunaHttp1Workspace26progress__auth__write__one(' \
  'LunaHttp1Workspace27progress__auth__finish__one(' \
  'LunaHttp1Workspace25progress__auth__work__one(' \
  'LunaHttp1Workspace27progress__auth__decide__one(' \
  'LunaHttp1View18require__workspace(' \
  'LunaHttp1View5route(' \
  'LunaHttp1View14body__byte__at(' \
  'LunaHttp1View7release(' \
  'LunaApiAuthVerifierWorkspace13begin__verify(' \
  'LunaApiAuthVerificationWrite10push__byte(' \
  'LunaApiAuthVerificationWrite6finish(' \
  'LunaApiAuthVerificationWork8progress(' \
  'LunaApiAuthVerifierWorkspace13progress__one(' \
  'LunaApiAuthPolicy13compare__byte(' \
  'LunaApiAuthVerificationWork14take__decision(' \
  'LunaApiAuthVerificationWrite5abort(' \
  'LunaApiAuthVerificationWork5abort(' \
  'LunaHttp1ResponseWorkspace11begin__kind(' \
  'LunaHttp1ResponseWorkspace20begin__event__stream(' \
  'LunaHttp1ResponseWorkspace12begin__error(' \
  'LunaHttp1ResponseWork18require__workspace(' \
  'LunaHttp1ResponseWork8progress(' \
  'LunaHttp1ResponseWork10take__view(' \
  'LunaHttp1ResponseWork5abort(' \
  'LunaHttp1ResponseView18require__workspace(' \
  'LunaHttp1ResponseView8byte__at(' \
  'LunaHttp1ResponseView11copy__chunk(' \
  'LunaHttp1ResponseView7release('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna HTTP/1 generated function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_success_allocation_or_copy; then
    printf 'Luna HTTP/1 warmed function allocates or copies: %s\n' "$symbol" >&2
    exit 1
  fi
done

production_files=(
  service/http1/types.mbt
  service/http1/workspace.mbt
  service/http1/progress.mbt
  service/http1/head_scan.mbt
  service/http1/head_validate.mbt
  service/http1/auth_progress.mbt
  service/http1/work_progress.mbt
  service/http1/view.mbt
  service/http1/response_types.mbt
  service/http1/response_workspace.mbt
  service/http1/response_progress.mbt
  service/http1/response_challenge.mbt
  service/http1/response_work.mbt
  service/http1/response_view.mbt
)
if rg -q 'String|Bytes|@utf8|ArrayView|ReadOnlyArray' "${production_files[@]}"; then
  printf '%s\n' 'Luna HTTP/1 production source materializes a payload object' >&2
  exit 1
fi
if ! rg -F -q "storage: FixedArray::make(cells, b'\\x00')" \
    service/http1/workspace.mbt ||
  ! rg -F -q 'auth_writes: Array::new(capacity=1)' \
    service/http1/workspace.mbt ||
  ! rg -F -q 'auth_works: Array::new(capacity=1)' \
    service/http1/workspace.mbt ||
  ! rg --pcre2 -U -q \
    '(?s)if self\.terminator_state == 4 \{.*?self\.phase = LUNA_HTTP1_SCANNING_LINE' \
    service/http1/progress.mbt ||
  ! rg --pcre2 -U -q \
    '(?s)if decision\.is_rejected\(\) \{.*?LunaHttp1Rejected.*?if self\.body_length == 0.*?else if self\.input_ended.*?else \{.*?self\.phase = LUNA_HTTP1_RECEIVING_BODY' \
    service/http1/auth_progress.mbt ||
  ! rg --pcre2 -U -q \
    '(?s)while consumed < length.*?workspace\.phase == LUNA_HTTP1_RECEIVING_HEAD \|\|.*?workspace\.phase == LUNA_HTTP1_RECEIVING_BODY' \
    service/http1/progress.mbt ||
  ! rg --pcre2 -U -q \
    '(?s)while used < budget && workspace\.phase == LUNA_HTTP1_RESPONSE_BUILDING.*?luna_http1_response_byte.*?workspace\.phase = LUNA_HTTP1_RESPONSE_FINALIZING' \
    service/http1/response_work.mbt ||
  ! rg --pcre2 -U -q \
    '(?s)let amount = Int::min\(.*?workspace\.budget\.work_units\(\).*?for index in 0..<amount' \
    service/http1/response_view.mbt; then
  printf '%s\n' 'Luna HTTP/1 storage or auth-before-body source proof drifted' >&2
  exit 1
fi

for type in \
  LunaHttp1Limits \
  LunaHttp1StepBudget \
  LunaHttp1Workspace \
  LunaHttp1Work \
  LunaHttp1View \
  LunaHttp1ResponseStepBudget \
  LunaHttp1ResponseWorkspace \
  LunaHttp1ResponseWork \
  LunaHttp1ResponseView; do
  if ! rg -U -q "pub struct ${type} \\{\\n  // private fields\\n\\}" "$mbti"; then
    printf 'Luna HTTP/1 type is not opaque: %s\n' "$type" >&2
    exit 1
  fi
done
if rg -q 'impl Debug for LunaHttp1(Limits|StepBudget|Workspace|Work|View|ResponseStepBudget|ResponseWorkspace|ResponseWork|ResponseView)' "$mbti" ||
  rg -q -- 'LunaHttp1.*-> .*(FixedArray|Array|ReadOnlyArray|ArrayView|Bytes|String)' "$mbti" ||
  rg -q 'LunaHttp1(View|Work|Workspace).*(Bytes|String|ArrayView|ReadOnlyArray)' "$mbti"; then
  printf '%s\n' 'Luna HTTP/1 authority or raw payload escaped' >&2
  exit 1
fi
if ! rg -U -q \
  'suberror LunaHttp1Error \{\n  LunaHttp1Failed\(rule~ : LunaHttp1Rule, issue~ : LunaHttp1Issue\)\n\}' \
  "$mbti" ||
  [ "$(rg -c '^pub fn LunaHttp1Limits::' "$mbti")" -ne 1 ] ||
  [ "$(rg -c '^pub fn LunaHttp1StepBudget::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaHttp1Workspace::' "$mbti")" -ne 5 ] ||
  [ "$(rg -c '^pub fn LunaHttp1Work::' "$mbti")" -ne 9 ] ||
  [ "$(rg -c '^pub fn LunaHttp1View::' "$mbti")" -ne 5 ] ||
  [ "$(rg -c '^pub fn LunaHttp1ResponseStepBudget::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaHttp1ResponseWorkspace::' "$mbti")" -ne 4 ] ||
  [ "$(rg -c '^pub fn LunaHttp1ResponseWork::' "$mbti")" -ne 6 ] ||
  [ "$(rg -c '^pub fn LunaHttp1ResponseView::' "$mbti")" -ne 6 ]; then
  printf '%s\n' 'Luna HTTP/1 exact public surface drifted' >&2
  exit 1
fi
if ! rg -U -q \
  'for index in 0\.\.<self\.storage\.length\(\) \{\n    self\.storage\[index\] = b'"'"'\\x00'"'"'' \
  service/http1/workspace.mbt ||
  ! rg -q 'fn LunaHttp1Workspace::wipe_credential' \
    service/http1/workspace.mbt ||
  ! rg -U -q \
    'for index in 0\.\.<self\.credential_length \{\n      self\.storage\[self\.credential_offset \+ index\] = b'"'"'\\x00'"'"'' \
    service/http1/workspace.mbt ||
  ! rg -q 'self\.wipe_credential\(\)' service/http1/workspace.mbt; then
  printf '%s\n' 'Luna HTTP/1 credential and terminal wipe boundary drifted' >&2
  exit 1
fi
if rg -q 'let retained_(head|body) = Int::min' service/http1/workspace.mbt; then
  printf '%s\n' 'Luna HTTP/1 request reuse regressed to full-buffer wiping' >&2
  exit 1
fi
for signature in \
  'pub fn LunaHttp1Workspace::close_authentication(Self) -> Unit' \
  'pub fn LunaHttp1Work::offer(Self, FixedArray[Byte], source_offset~ : Int, length~ : Int) -> Int raise LunaHttp1Error' \
  'pub fn LunaHttp1Work::take_view(Self) -> LunaHttp1View raise LunaHttp1Error' \
  'pub fn LunaHttp1View::body_byte_at(Self, Int) -> Byte raise LunaHttp1Error' \
  'pub fn LunaHttp1View::body_length(Self) -> Int raise LunaHttp1Error' \
  'pub fn LunaHttp1View::is_live(Self) -> Bool' \
  'pub fn LunaHttp1View::release(Self) -> Unit raise LunaHttp1Error' \
  'pub fn LunaHttp1View::route(Self) -> LunaHttp1Route raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseWorkspace::begin_error(Self, LunaHttp1ErrorResponseKind) -> LunaHttp1ResponseWork raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseWorkspace::begin_event_stream(Self) -> LunaHttp1ResponseWork raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseView::byte_at(Self, Int) -> Byte raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseView::copy_chunk(Self, FixedArray[Byte], destination_offset~ : Int, source_offset~ : Int, length~ : Int) -> Int raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseView::is_live(Self) -> Bool' \
  'pub fn LunaHttp1ResponseView::length(Self) -> Int raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseView::release(Self) -> Unit raise LunaHttp1Error' \
  'pub fn LunaHttp1ResponseView::status_code(Self) -> Int raise LunaHttp1Error'; do
  if ! rg -F -x -q "$signature" "$mbti"; then
    printf 'Luna HTTP/1 exact signature drifted: %s\n' "$signature" >&2
    exit 1
  fi
done
if ! rg -U -q \
  'pub\(all\) enum LunaHttp1ErrorResponseKind \{\n  LunaHttp1AuthenticationError\n  LunaHttp1ValidationError\n  LunaHttp1CapacityError\n  LunaHttp1ServiceError\n\}' \
  "$mbti"; then
  printf '%s\n' 'Luna HTTP/1 fixed error vocabulary drifted' >&2
  exit 1
fi

printf '%s\n' 'Luna HTTP/1 allocation, ownership, and source gate passed.'
