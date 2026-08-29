#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
moon test service/request_admission --target native --release --deny-warn --warn-list +73

generated="_build/native/release/test/service/request_admission/request_admission.whitebox_test.c"
if [ ! -f "$generated" ]; then
  printf '%s\n' 'Luna trusted receipt release C is missing' >&2
  exit 1
fi

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
  ' "$generated"
}

forbidden='moonbit_malloc|moonbit_make_|moonbit_add_string|memcpy|memmove'
for symbol in \
  'LunaRequestReceiptWorkspace5begin(' \
  'LunaRequestReceipt18require__workspace(' \
  'LunaRequestReceipt7observe(' \
  'LunaRequestReceipt17remaining__millis(' \
  'LunaRequestReceipt7consume(' \
  'LunaRequestReceipt5abort('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna trusted receipt function missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | rg "$forbidden" |
    rg -v 'moonbit_malloc.*RequestAdmissionError_2e(Invalid|ClockUnavailable)' |
    rg -q .; then
    printf 'Luna trusted receipt warmed scalar path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

positive="$(extract_definition 'new__luna__receipt__workspace__with__clock(')"
if [ -z "$positive" ] || ! printf '%s\n' "$positive" | rg -q 'moonbit_malloc'; then
  printf '%s\n' 'Luna trusted receipt allocation positive control failed' >&2
  exit 1
fi

mbti="service/request_admission/pkg.generated.mbti"
for type in LunaRequestReceiptWorkspace LunaRequestReceipt; do
  if ! rg -U -q "pub struct ${type} \{\n  // private fields\n\}" "$mbti" ||
    rg -q "impl Debug for ${type}" "$mbti"; then
    printf 'Luna trusted receipt authority surface drifted: %s\n' "$type" >&2
    exit 1
  fi
done
if [ "$(rg -c '^pub fn LunaRequestReceiptWorkspace::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaRequestReceipt::' "$mbti")" -ne 2 ] ||
  [ "$(rg -c '^pub fn LunaRequestPreparationPool::try_begin_luna_framed_with_receipt' "$mbti")" -ne 1 ] ||
  rg -q 'LunaRequestReceipt.*(received_at|deadline_at|budget_millis|UInt64)' "$mbti"; then
  printf '%s\n' 'Luna trusted receipt public API drifted' >&2
  exit 1
fi
if ! rg -U -q 'budget_millis > self\.inference\.max_deadline_millis\(\)[\s\S]*receipt\.observe\(\)[\s\S]*framed_workspace\.begin\(\)[\s\S]*receipt\.consume\(\)' \
    service/request_admission/luna_trusted_receipt_pool.mbt; then
  printf '%s\n' 'Luna trusted receipt preflight/transfer ordering drifted' >&2
  exit 1
fi

printf '%s\n' 'Luna trusted receipt allocation and API gate passed.'
