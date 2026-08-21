#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test contracts/inference \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/contracts/inference/inference.blackbox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna request-semantics release C output is missing' >&2
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

allocation_lines() {
  rg 'moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string' || true
}

# Storage construction is the positive control: the reusable arrays must be
# allocated at startup before bounded semantic Work begins.
storage_new="$(extract_definition 'LunaRequestSemanticStorage3new(')"
if [ -z "$storage_new" ] ||
  ! printf '%s\n' "$storage_new" | allocation_lines | rg -q .; then
  printf '%s\n' 'Luna request-semantics allocation positive control is ineffective' >&2
  exit 1
fi

# These are the complete noraise core of one authenticated membership lookup
# and every charged semantic Work transition. The public Work wrapper can
# allocate a checked-error shell only on stale authority, so it is deliberately
# excluded rather than hidden behind an error-line filter.
for symbol in \
  'LunaRequestSemanticView15is__stop__token(' \
  'LunaRequestSemanticStorage13progress__one(' \
  'LunaRequestSemanticStorage22progress__token__value(' \
  'LunaRequestSemanticStorage26progress__token__duplicate(' \
  'LunaRequestSemanticStorage24progress__string__header(' \
  'LunaRequestSemanticStorage22progress__string__utf8(' \
  'LunaRequestSemanticStorage35progress__string__duplicate__length(' \
  'LunaRequestSemanticStorage33progress__string__duplicate__byte(' \
  'LunaRequestSemanticStorage23progress__cache__header(' \
  'LunaRequestSemanticStorage21progress__cache__byte(' \
  'LunaRequestSemanticStorage20progress__sort__copy(' \
  'LunaRequestSemanticStorage20progress__sort__load(' \
  'LunaRequestSemanticStorage23progress__sort__compare(' \
  'LunaRequestSemanticStorage20fail__string__format(' \
  'LunaRequestSemanticStorage14fail__semantic(' \
  '26is__safe__identifier__byte('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna request-semantics allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | allocation_lines | rg -q .; then
    printf 'Luna request-semantics bounded core allocates: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux request-semantics allocation gate passed.'
