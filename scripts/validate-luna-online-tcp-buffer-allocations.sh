#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/online_tcp \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/online_tcp/online_tcp.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna online TCP scratch release C output is missing' >&2
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

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|Bytes5makei|memcpy|memmove|blit'

contains_forbidden_allocation_or_copy() {
  # Exclude only MoonBit's exact private typed-error envelope. Broad
  # Error/Failure-name filtering would turn this into a false proof.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

constructor_body="$(extract_definition 'LunaOnlineTcpOutputScratch3new(')"
if [ -z "$constructor_body" ] ||
  [ "$(printf '%s\n' "$constructor_body" |
    rg -c 'moonbit_make_bytes\(')" -ne 1 ] ||
  [ "$(printf '%s\n' "$constructor_body" |
    rg -c 'retain__bytes__as__fixed__array\(')" -ne 1 ]; then
  printf '%s\n' 'online TCP scratch allocation positive control is ineffective' >&2
  exit 1
fi

if [ "$(rg -c 'Bytes::makei\(' service/online_tcp/scratch.mbt)" -ne 1 ] ||
  rg -n 'Bytes::(make|makei|from|to_owned)|FixedArray::make|Array::|BytesView|memcpy|memmove|blit' \
    service/online_tcp/scratch.mbt service/online_tcp/scratch_types.mbt |
    rg -v 'scratch\.mbt:9:  let immutable = Bytes::makei'; then
  printf '%s\n' \
    'online TCP scratch must own exactly one startup Bytes backing' >&2
  exit 1
fi

for symbol in \
  'LunaOnlineTcpOutputScratch12begin__write(' \
  'LunaOnlineTcpOutputWrite17require__mutating(' \
  'LunaOnlineTcpOutputWrite11write__byte(' \
  'LunaOnlineTcpOutputWrite10copy__from(' \
  'LunaOnlineTcpOutputWrite7publish(' \
  'LunaOnlineTcpOutputWrite5abort(' \
  'LunaOnlineTcpOutputFlight19require__in__flight(' \
  'LunaOnlineTcpOutputFlight12byte__length(' \
  'LunaOnlineTcpOutputFlight8byte__at(' \
  'LunaOnlineTcpOutputFlight7release('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'online TCP scratch allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation_or_copy; then
    printf 'online TCP warmed payload-buffer path allocates or bulk-copies: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

if ! rg -q --pcre2 -U \
    "let immutable = Bytes::makei\\(capacity, _ => b'\\\\x00'\\)\\n  let mutable = @buffer_alias\\.retain_bytes_as_fixed_array\\(immutable\\)" \
    service/online_tcp/scratch.mbt ||
  [ "$(rg -c '@buffer_alias\.retain_bytes_as_fixed_array\(' \
    service/online_tcp/scratch.mbt)" -ne 1 ]; then
  printf '%s\n' \
    'online TCP scratch alias construction stopped being one startup operation' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux online TCP warmed payload-buffer allocation/copy gate passed.'
