#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73
moon test engine/worker_service \
  --target native --release --deny-warn --warn-list +73
scripts/validate-hot-path-allocations.sh
scripts/validate-worker-process-exchange-allocations.sh

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'worker-service release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^struct moonbit_result_/ {
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

hot_body=""
for symbol in \
  'WorkerService8progress(' \
  'WorkerService16commit__received('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'worker-service allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string'; then
  printf '%s\n' 'worker-service progress constructs a collection, ref, or string' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'Error|Failure'; then
  printf '%s\n' 'worker-service progress contains a non-error heap allocation' >&2
  exit 1
fi

if ! rg -q 'Bytes4make\(17, 81\)' \
  _build/native/release/test/engine/worker_service/worker_service.whitebox_test.c; then
  printf '%s\n' 'worker-service allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker-service progress allocation gate passed.'
