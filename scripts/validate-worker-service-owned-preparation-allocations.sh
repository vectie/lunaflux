#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon build tests/worker_service_e2e \
  --target native --release --deny-warn --warn-list +73
moon test engine/worker_service \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'owned worker-service release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^[A-Za-z_][A-Za-z0-9_ *]*_M0/ {
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

forbidden='moonbit_malloc|moonbit_make_|Bytes4make'
positive_c="_build/native/release/test/engine/worker_service/worker_service.whitebox_test.c"
positive_body="$(awk '
  /owned__preparation__allocation__positive__control\(/ {
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
' "$positive_c")"
if [ -z "$positive_body" ] || ! printf '%s\n' "$positive_body" | rg -q "$forbidden"; then
  printf '%s\n' 'owned preparation allocation positive control is ineffective' >&2
  exit 1
fi

for symbol in \
  'install__owned__ready(' \
  'publish__cleanup(' \
  'prepare__preflighted(' \
  'prepare__spawned(' \
  'perform__startup__handshake__status(' \
  'spawn__with__approved__roots__status(' \
  'startup__write__frame__status(' \
  'startup__read__frame__status(' \
  'startup__close__status(' \
  'close__root__pair__status(' \
  'root__failure__status(' \
  'roots__still__live(' \
  'acquire__prepared__worker__approved__roots__status('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'owned preparation publication function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | rg -q "$forbidden"; then
    printf 'owned preparation publication allocates after rooted startup: %s\n' "$symbol" >&2
    exit 1
  fi
done

for symbol in \
  'publish__ready__root__bound(' \
  'publish__live__root__bound__cleanup('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ] || printf '%s\n' "$body" | rg -q "$forbidden"; then
    printf 'rooted post-activation publication allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

prepare_line="$(rg -n 'prepare__exchange__with__approved__roots\(' "$generated_c" | tail -n 1 | cut -d: -f1)"
service_line="$(rg -n 'WorkerService\*\)moonbit_malloc' "$generated_c" | tail -n 1 | cut -d: -f1)"
cleanup_line="$(rg -n 'FailedOwnedWorkerServicePreparation\*\)moonbit_malloc' "$generated_c" | tail -n 1 | cut -d: -f1)"
outcome_line="$(rg -n 'OwnedWorkerServicePreparation\*\)moonbit_malloc' "$generated_c" | tail -n 1 | cut -d: -f1)"
if [ -z "$prepare_line" ] || [ -z "$service_line" ] ||
  [ -z "$cleanup_line" ] || [ -z "$outcome_line" ] ||
  [ "$service_line" -ge "$prepare_line" ] ||
  [ "$cleanup_line" -ge "$prepare_line" ] ||
  [ "$outcome_line" -ge "$prepare_line" ]; then
  printf '%s\n' 'owned service/cleanup shells were not allocated before rooted preparation' >&2
  exit 1
fi

root_body="$(extract_definition 'prepare__root__bound(')"
if [ -z "$root_body" ]; then
  printf '%s\n' 'root-bound preparation definition is missing' >&2
  exit 1
fi
root_activation_line="$(printf '%s\n' "$root_body" | rg -n 'acquire__prepared__worker__approved__roots__status\(' | cut -d: -f1)"
root_ready_line="$(printf '%s\n' "$root_body" | rg -n 'prepare__preflighted\(' | cut -d: -f1)"
root_ready_owner_line="$(printf '%s\n' "$root_body" | rg -n 'RootBoundWorkerProcessSupervisor\*\)moonbit_malloc' | cut -d: -f1)"
root_failed_owner_line="$(printf '%s\n' "$root_body" | rg -n 'FailedRootBoundWorkerProcessStartup\*\)moonbit_malloc' | cut -d: -f1)"
if [ -z "$root_activation_line" ] || [ -z "$root_ready_line" ] ||
  [ -z "$root_ready_owner_line" ] || [ -z "$root_failed_owner_line" ] ||
  [ "$root_ready_owner_line" -ge "$root_activation_line" ] ||
  [ "$root_failed_owner_line" -ge "$root_activation_line" ] ||
  [ "$root_ready_line" -le "$root_activation_line" ]; then
  printf '%s\n' 'root-bound shells/order do not dominate root activation and startup' >&2
  exit 1
fi
if [ "$root_ready_line" -le "$root_activation_line" ]; then
  printf '%s\n' 'scalar rooted startup must follow root activation' >&2
  exit 1
fi
if printf '%s\n' "$root_body" | rg -q \
  '(RootBoundWorkerProcessPreparation|WorkerProcessPreparation)[^\n]*moonbit_malloc'; then
  printf '%s\n' 'rooted preparation result enums must remain inline' >&2
  exit 1
fi

# The direct dispatcher between scalar startup and the two allocation-free
# publication helpers must also remain allocation-free. Closed/no-authority
# diagnostic construction follows these live-authority branch points and is
# deliberately outside these slices.
startup_call_line="$(printf '%s\n' "$root_body" | rg -n 'prepare__preflighted\(' | tail -n 1 | cut -d: -f1)"
ready_publish_line="$(printf '%s\n' "$root_body" | rg -n 'publish__ready__root__bound\(' | tail -n 1 | cut -d: -f1)"
root_close_line="$(printf '%s\n' "$root_body" | rg -n 'close__root__pair__status\(' | tail -n 1 | cut -d: -f1)"
live_publish_line="$(printf '%s\n' "$root_body" | rg -n 'publish__live__root__bound__cleanup\(' | tail -n 1 | cut -d: -f1)"
ready_return_line="$(printf '%s\n' "$root_body" | awk -v start="$ready_publish_line" \
  'NR > start && /return _result_/ { print NR; exit }')"
live_return_line="$(printf '%s\n' "$root_body" | awk -v start="$live_publish_line" \
  'NR > start && /return _result_/ { print NR; exit }')"
cleanup_branch_line="$((ready_return_line + 1))"
if [ -z "$startup_call_line" ] || [ -z "$ready_publish_line" ] ||
  [ -z "$root_close_line" ] || [ -z "$live_publish_line" ] ||
  [ -z "$ready_return_line" ] || [ -z "$live_return_line" ] ||
  [ "$ready_return_line" -le "$ready_publish_line" ] ||
  [ "$root_close_line" -le "$cleanup_branch_line" ] ||
  [ "$live_return_line" -le "$live_publish_line" ]; then
  printf '%s\n' 'rooted post-activation dispatcher branch is missing' >&2
  exit 1
fi
for branch in \
  "${startup_call_line},${ready_return_line}" \
  "${cleanup_branch_line},${live_return_line}"; do
  branch_body="$(printf '%s\n' "$root_body" | sed -n "${branch}p")"
  if printf '%s\n' "$branch_body" | rg -q "$forbidden"; then
    printf '%s\n' 'direct rooted post-activation dispatch allocates' >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux owned worker-service rooted startup allocation gate passed.'
