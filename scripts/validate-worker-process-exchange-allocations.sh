#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test engine/worker_process \
  --target native --release --deny-warn --warn-list +73
moon test internal/process \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/engine/worker_process/worker_process.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'worker-process release C output is missing' >&2
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
  'RootBoundWorkerProcessSupervisor15begin__exchange(' \
  'RootBoundWorkerProcessSupervisor18progress__exchange(' \
  'WorkerProcessSupervisor22begin__exchange__write(' \
  'WorkerProcessSupervisor35accept__exchange__completion__bytes('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'worker exchange allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string'; then
  printf '%s\n' 'worker exchange path constructs a collection, ref, or string' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'Error|Failure'; then
  printf '%s\n' 'worker exchange path contains a non-error heap allocation' >&2
  exit 1
fi

# Positive control shared with the internal owner-resident transfer gate.
process_c="_build/native/release/test/internal/process/process.whitebox_test.c"
if [ ! -f "$process_c" ] || ! rg -q 'moonbit_make_bytes\(13, 90\)' "$process_c"; then
  printf '%s\n' 'worker exchange allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker exchange allocation gate passed.'
