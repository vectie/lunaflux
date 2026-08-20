#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/incremental_output \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/incremental_output/incremental_output.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'incremental-output release C output is missing' >&2
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

contains_unbounded_allocation() {
  allocation_lines |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

# Construction is the positive control: reusable workspace storage must be
# allocated before the warmed progress/token path begins.
workspace_new="$(extract_definition '41new__luna__incremental__output__workspace(')"
if [ -z "$workspace_new" ] ||
  ! printf '%s\n' "$workspace_new" | allocation_lines | rg -q .; then
  printf '%s\n' 'incremental-output allocation positive control is ineffective' >&2
  exit 1
fi

# Scan the actual reusable setup and token-step closure. Missing emitted
# symbols are failures, so inlining cannot silently narrow the evidence.
for symbol in \
  'LunaIncrementalOutputWorkspace5begin(' \
  'LunaIncrementalOutputWork8progress(' \
  'LunaIncrementalOutputWorkspace20progress__setup__one(' \
  'LunaIncrementalOutputWorkspace21progress__setup__text(' \
  'LunaIncrementalOutputWorkspace24progress__setup__pattern(' \
  'LunaIncrementalOutputWorkspace24progress__setup__failure(' \
  'LunaIncrementalOutputWork11take__lease(' \
  'LunaIncrementalOutputLease19push__token__status(' \
  'LunaIncrementalOutputLease35push__token__without__stops__status(' \
  'LunaIncrementalOutputLease20finish__into__status(' \
  'LunaIncrementalOutputWorkspace28push__token__status__current(' \
  '16pattern__advance(' \
  'TokenizerSpec28copy__decoded__piece__status(' \
  'LunaIncrementalOutputLease7release('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'incremental-output allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_unbounded_allocation; then
    printf 'incremental-output warmed path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux incremental-output allocation gate passed.'
