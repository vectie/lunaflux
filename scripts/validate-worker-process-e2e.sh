#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

moon build cmd/worker_echo cmd/worker_echo_fail tests/worker_process_e2e tests/worker_service_e2e \
  --target native --release --deny-warn

worker="$repo_root/_build/native/release/build/cmd/worker_echo/worker_echo.exe"
gate="$repo_root/_build/native/release/build/tests/worker_process_e2e/worker_process_e2e.exe"
service_gate="$repo_root/_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.exe"
failure_worker="$repo_root/_build/native/release/build/cmd/worker_echo_fail/worker_echo_fail.exe"

if [ ! -x "$worker" ] || [ ! -x "$failure_worker" ] ||
  [ ! -x "$gate" ] || [ ! -x "$service_gate" ]; then
  printf '%s\n' 'worker-process end-to-end executables are missing' >&2
  exit 1
fi

"$gate" "$worker"
"$service_gate" "$worker" "$failure_worker"
printf '%s\n' 'LunaFlux worker process end-to-end gate passed.'
