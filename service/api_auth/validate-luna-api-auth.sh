#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

moon test service/api_auth --target native --release --deny-warn

generated_c_files=(
  "_build/native/release/test/service/api_auth/api_auth.whitebox_test.c"
  "_build/native/release/test/service/api_auth/api_auth.blackbox_test.c"
)
for generated_c in "${generated_c_files[@]}"; do
  if [ ! -f "$generated_c" ]; then
    printf 'Luna API authentication release C output is missing: %s\n' \
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
    rg -v 'moonbit_malloc.*LunaApiAuthError_2eLunaApiAuthFailed' |
    rg -q .
}

positive="$(extract_definition 'LunaApiAuthPolicy3new(')"
if [ -z "$positive" ] ||
  ! printf '%s\n' "$positive" | rg -q "$forbidden"; then
  printf '%s\n' 'Luna API authentication allocation positive control failed' >&2
  exit 1
fi

for symbol in \
  'LunaApiAuthPolicy5close(' \
  'LunaApiAuthPolicy6verify(' \
  'LunaApiAuthPolicy24luna__api__auth__compare(' \
  'LunaApiAuthPolicy13compare__byte(' \
  'LunaApiAuthPolicy17is__authenticated(' \
  'api__auth33luna__api__auth__range__is__valid(' \
  'LunaApiAuthVerifierWorkspace13begin__verify(' \
  'LunaApiAuthVerifierWorkspace5close(' \
  'LunaApiAuthVerificationWrite10push__byte(' \
  'LunaApiAuthVerificationWork8progress(' \
  'LunaApiAuthVerifierWorkspace13progress__one('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna API authentication function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_success_allocation_or_copy; then
    printf 'Luna API authentication warmed path allocates or copies: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

if ! rg -U -q \
  'for index in 0\.\.<self\.expected\.length\(\) \{\n    self\.expected\[index\] = b'"'"'\\x00'"'"'' \
  service/api_auth/policy.mbt ||
  ! rg -q 'self\.maximum_credential_bytes = 0' service/api_auth/policy.mbt ||
  ! rg -q 'self\.phase = LUNA_API_AUTH_CLOSED' service/api_auth/verifier.mbt; then
  printf '%s\n' 'Luna API authentication deterministic wipe drifted' >&2
  exit 1
fi

mbti="service/api_auth/pkg.generated.mbti"
for type in \
  LunaApiAuthPolicy \
  LunaApiAuthDecision \
  LunaApiAuthStepBudget \
  LunaApiAuthWriteProgress \
  LunaApiAuthVerifierWorkspace \
  LunaApiAuthVerificationWrite \
  LunaApiAuthVerificationWork; do
  if ! rg -U -q "(pub )?struct ${type} \\{\\n  // private fields\\n\\}" "$mbti"; then
    printf 'Luna API authentication type is not opaque: %s\n' "$type" >&2
    exit 1
  fi
done
if rg -q 'impl Debug for LunaApiAuth(Policy|Decision|StepBudget|WriteProgress|VerifierWorkspace|VerificationWrite|VerificationWork)' "$mbti" ||
  rg -q 'LunaApiAuth(Policy|Decision|StepBudget|WriteProgress|VerifierWorkspace|VerificationWrite|VerificationWork).*(String|Bytes|ArrayView|ReadOnlyArray)' "$mbti" ||
  rg -q -- 'LunaApiAuth.*-> (FixedArray|Array|ReadOnlyArray|ArrayView|Bytes|String)' "$mbti"; then
  printf '%s\n' 'Luna API authentication authority surface leaked' >&2
  exit 1
fi
if ! rg -U -q \
  'suberror LunaApiAuthError \{\n  LunaApiAuthFailed\(rule~ : LunaApiAuthRule, issue~ : LunaApiAuthIssue\)\n\}' \
  "$mbti"; then
  printf '%s\n' 'Luna API authentication error payload surface drifted' >&2
  exit 1
fi
if [ "$(rg -c '^pub fn LunaApiAuthPolicy::' "$mbti")" -ne 4 ] ||
  [ "$(rg -c '^pub fn LunaApiAuthDecision::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaApiAuthStepBudget::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaApiAuthWriteProgress::' "$mbti")" -ne 3 ] ||
  [ "$(rg -c '^pub fn LunaApiAuthVerifierWorkspace::' "$mbti")" -ne 3 ] ||
  [ "$(rg -c '^pub fn LunaApiAuthVerificationWrite::' "$mbti")" -ne 3 ] ||
  [ "$(rg -c '^pub fn LunaApiAuthVerificationWork::' "$mbti")" -ne 7 ]; then
  printf '%s\n' 'Luna API authentication method surface drifted' >&2
  exit 1
fi
for predicate in is_accepted is_rejected; do
  if ! rg -U -q \
    "pub fn LunaApiAuthDecision::${predicate}[^}]+self\\.state == [01]" \
    service/api_auth/types.mbt; then
    printf 'Luna API authentication predicate drifted: %s\n' "$predicate" >&2
    exit 1
  fi
done
if ! rg -q 'for index in 0..<self\.expected_length' \
  service/api_auth/policy.mbt ||
  ! rg -q 'difference = self\.compare_byte' service/api_auth/policy.mbt; then
  printf '%s\n' \
    'Luna API authentication configured-secret comparison loop drifted' >&2
  exit 1
fi
if rg -q 'for index in 0..<self\.maximum_credential_bytes' \
  service/api_auth/policy.mbt; then
  printf '%s\n' \
    'Luna API authentication regressed to maximum-bound padding work' >&2
  exit 1
fi

printf '%s\n' 'Luna API authentication allocation and source gate passed.'
