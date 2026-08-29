#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test tests/inference_credential_e2e \
  --target native --release --deny-warn

generated_c="_build/native/release/test/tests/inference_credential_e2e/inference_credential_e2e.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'credential release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    !copying && $0 ~ pattern {
      candidate = 1; body = $0 ORS; next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) {
        candidate = 0; body = ""; next
      }
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

hot_body=""
for symbol in \
  'InheritedInferenceCredentialOwner8progress' \
  'InheritedInferenceCredentialOwner12read__header' \
  'InheritedInferenceCredentialOwner16read__credential'; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'credential allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_malloc|moonbit_make_|moonbit_add_string'; then
  printf '%s\n' 'credential progress path contains a heap allocation' >&2
  exit 1
fi

if ! rg -q 'moonbit_make_bytes\(4096, 0\)' "$generated_c"; then
  printf '%s\n' 'credential allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux inference-credential progress allocation gate passed.'
