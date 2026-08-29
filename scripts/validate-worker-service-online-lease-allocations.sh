#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e --target native --release --deny-warn
moon test engine/worker_service --target native --release --deny-warn
moon test scheduler/core --target native --release --deny-warn

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
positive_c="_build/native/release/test/engine/worker_service/worker_service.whitebox_test.c"
scheduler_c="_build/native/release/test/scheduler/core/core.whitebox_test.c"

extract_definition() {
  local file="$1"
  local pattern="$2"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^(struct moonbit_result_|moonbit_bytes_t|int32_t|uint64_t)/ {
      candidate = 1; body = $0 ORS; next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) { copying = 1; depth = 1; printf "%s", body; candidate = 0; next }
    }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "$file"
}

forbidden='Bytes4make|moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string|moonbit_malloc'

control="$(extract_definition "$positive_c" 'online__lease__allocation__positive__control(')"
if [ -z "$control" ] || ! printf '%s\n' "$control" | rg -q "$forbidden"; then
  printf '%s\n' 'online-lease allocation positive control is ineffective' >&2
  exit 1
fi

hot=""
for symbol in \
  'OnlineWorkerLease17take__publication(' \
  'OnlineWorkerLease21take__reserved__token(' \
  'OnlineWorkerLease14commit__cancel(' \
  'OnlineWorkerLease8progress(' \
  'WorkerService32progress__existing__flight__impl(' \
  'OnlineWorkerLease6expire(' \
  'Scheduler29has__exact__natural__terminal(' \
  'Scheduler13expire__exact(' \
  'Scheduler26expire__exact__or__natural(' \
  'Scheduler17commit__admission(' \
  'Scheduler26commit__admission__request(' \
  'OnlineWorkerLease25invalidate__device__state(' \
  'Scheduler27advance__terminated__handle(' \
  'Scheduler25invalidate__device__state('; do
  body="$(extract_definition "$generated_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'online-lease allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot="${hot}${body}"
done

for symbol in \
  'OnlineWorkerLease25drain__restart__forbidden(' \
  'read__expiry__clockGRPC13ref3RefGiEE('; do
  body="$(extract_definition "$positive_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'online-lease whitebox allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot="${hot}${body}"
done

for symbol in \
  'read__expiry__clockGRP46vectie8lunaflux6engine15worker__service17OnlineWorkerLeaseE(' \
  'read__online__lease__clock('; do
  body="$(extract_definition "$generated_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'online expiry allocation helper is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot="${hot}${body}"
done

for symbol in \
  'Scheduler28drain__instance__loss__exact(' \
  'Scheduler32invalidate__device__state__exact('; do
  body="$(extract_definition "$scheduler_c" "$symbol")"
  if [ -z "$body" ]; then
    printf 'online recovery allocation function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot="${hot}${body}"
done

if printf '%s\n' "$hot" | rg "$forbidden" | rg -q -v 'Error|Failure'; then
  printf '%s\n' 'online-lease steady transition contains a non-error allocation' >&2
  exit 1
fi

if rg -q 'moonbit_malloc.*OnlineWorkerPublication' "$generated_c"; then
  printf '%s\n' 'sanitized online publication is boxed' >&2
  exit 1
fi

scripts/validate-worker-service-progress-allocations.sh
scripts/validate-scheduler-cancel-reservation-allocations.sh
printf '%s\n' 'LunaFlux worker-service online-lease allocation gate passed.'
