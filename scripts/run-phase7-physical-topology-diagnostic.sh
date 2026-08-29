#!/bin/sh

set -eu
umask 077

if [ "$#" -ne 1 ]; then
  printf '%s\n' 'usage: run-phase7-physical-topology-diagnostic.sh NEW_EVIDENCE_DIR' >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
evidence_dir=$1
if [ -e "$evidence_dir" ]; then
  printf '%s\n' 'evidence directory must not already exist' >&2
  exit 2
fi

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-phase7-topology.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

if ! (cd "$repo_root" && moon build cmd/phase7_topology_diagnostic \
  --target native --deny-warn) >"$task_dir/build.stdout" \
  2>"$task_dir/build.stderr"; then
  sed -n '1,160p' "$task_dir/build.stdout" >&2
  sed -n '1,160p' "$task_dir/build.stderr" >&2
  exit 1
fi
runner="$repo_root/_build/native/debug/build/cmd/phase7_topology_diagnostic/phase7_topology_diagnostic.exe"
if [ ! -x "$runner" ]; then
  printf '%s\n' 'physical topology diagnostic executable is missing' >&2
  exit 1
fi
mkdir "$evidence_dir"

set +e
"$runner" >"$evidence_dir/stdout.log" 2>"$evidence_dir/stderr.log"
exit_status=$?
set -e

if command -v sha256sum >/dev/null 2>&1; then
  runner_sha256=$(sha256sum "$runner" | awk '{print $1}')
  stdout_sha256=$(sha256sum "$evidence_dir/stdout.log" | awk '{print $1}')
  stderr_sha256=$(sha256sum "$evidence_dir/stderr.log" | awk '{print $1}')
else
  runner_sha256=$(shasum -a 256 "$runner" | awk '{print $1}')
  stdout_sha256=$(shasum -a 256 "$evidence_dir/stdout.log" | awk '{print $1}')
  stderr_sha256=$(shasum -a 256 "$evidence_dir/stderr.log" | awk '{print $1}')
fi
runner_schema=$(sed -n 's/^schema: //p' "$evidence_dir/stdout.log")
outcome=$(sed -n 's/^outcome: //p' "$evidence_dir/stdout.log")

{
  printf '%s\n' 'evidence_schema=lunaflux.phase7.physical_topology.capture.v1'
  printf 'runner_schema=%s\n' "$runner_schema"
  printf 'exit_status=%s\n' "$exit_status"
  printf 'outcome=%s\n' "$outcome"
  printf 'runner_sha256=%s\n' "$runner_sha256"
  printf 'stdout_sha256=%s\n' "$stdout_sha256"
  printf 'stderr_sha256=%s\n' "$stderr_sha256"
} >"$evidence_dir/RESULT.txt"

valid=true
[ "$exit_status" -eq 0 ] || valid=false
[ "$runner_schema" = 'lunaflux.phase7.physical_topology.v1' ] || valid=false
case "$outcome" in supported|rejected) ;; *) valid=false ;; esac
[ ! -s "$evidence_dir/stderr.log" ] || valid=false
schema_count=$(sed -n '/^schema: /p' "$evidence_dir/stdout.log" | wc -l | tr -d ' ')
outcome_count=$(sed -n '/^outcome: /p' "$evidence_dir/stdout.log" | wc -l | tr -d ' ')
[ "$schema_count" -eq 1 ] || valid=false
[ "$outcome_count" -eq 1 ] || valid=false

chmod 0444 "$evidence_dir/stdout.log" "$evidence_dir/stderr.log" \
  "$evidence_dir/RESULT.txt"
chmod 0555 "$evidence_dir"

if [ "$valid" != true ]; then
  printf '%s\n' 'physical topology diagnostic evidence is invalid' >&2
  exit 1
fi
printf '%s\n' "$evidence_dir"
