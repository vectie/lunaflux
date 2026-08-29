#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/benchmark-campaign-common.sh"

[ "$#" -eq 10 ] || {
  printf '%s\n' \
    'usage: validate-phase4-prefix-benchmark.sh ABSOLUTE_CAMPAIGN PHASE4_POLICY#sha256=HEX PHASE4_GATE#sha256=HEX PHASE4_FD_EXEC_HELPER#sha256=HEX CAMPAIGN_TOOL#sha256=HEX TRIAL_DRIVER#sha256=HEX CORRECTNESS_VERIFIER#sha256=HEX ENGINE_IDENTITY_VERIFIER#sha256=HEX PROCESS_SUPERVISOR#sha256=HEX AUTHORITY' >&2
  exit 2
}

campaign_root=$1
policy_argument=$2
gate_argument=$3
helper_argument=$4
campaign_tool=$5
trial_driver=$6
correctness_verifier=$7
identity_verifier=$8
process_supervisor=$9
authority=${10}

[ "$authority" = phase4-measurement-gate-only ] ||
  benchmark_campaign_fail 'Phase 4 authority is not the exact measurement-only label'

case "$campaign_root" in
  /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
    benchmark_campaign_fail 'campaign root is not a safe canonical path'
    ;;
  /*) ;;
  *) benchmark_campaign_fail 'campaign root is not absolute' ;;
esac
[ -d "$campaign_root" ] && [ ! -L "$campaign_root" ] ||
  benchmark_campaign_fail 'campaign root is not a non-symlink directory'
[ "$(CDPATH= cd -- "$campaign_root" && pwd -P)" = "$campaign_root" ] ||
  benchmark_campaign_fail 'campaign root contains a directory alias'

phase4_split_identity() {
  case "$1" in
    /*#sha256=*) ;;
    *) benchmark_campaign_fail "$2 is not an absolute digest-suffixed path" ;;
  esac
  phase4_identity_path=${1%#sha256=*}
  phase4_identity_sha=${1##*#sha256=}
  benchmark_campaign_require_file "$phase4_identity_path" "$2"
  benchmark_campaign_require_digest \
    "$phase4_identity_path" "$phase4_identity_sha" "$2"
}

phase4_split_identity "$policy_argument" 'Phase 4 policy'
policy_path=$phase4_identity_path
policy_sha=$phase4_identity_sha
phase4_split_identity "$gate_argument" 'Phase 4 gate executable'
gate_path=$phase4_identity_path
gate_sha=$phase4_identity_sha
[ -x "$gate_path" ] ||
  benchmark_campaign_fail 'Phase 4 gate executable is not executable'
phase4_split_identity "$helper_argument" 'Phase 4 FD execution helper'
helper_path=$phase4_identity_path
helper_sha=$phase4_identity_sha
[ -x "$helper_path" ] ||
  benchmark_campaign_fail 'Phase 4 FD execution helper is not executable'

benchmark_campaign_require_size "$policy_path" 8192 'Phase 4 policy'
benchmark_campaign_require_newline "$policy_path" 'Phase 4 policy'
[ "$(wc -l < "$policy_path" | tr -d ' ')" -eq 39 ] ||
  benchmark_campaign_fail 'Phase 4 policy has the wrong shape'
[ "$(benchmark_campaign_field "$policy_path" 1 schema)" = \
    lunaflux.phase4-prefix-benchmark-policy.v1 ] ||
  benchmark_campaign_fail 'Phase 4 policy schema is invalid'

phase4_split_identity "$campaign_tool" 'campaign tool'
campaign_tool_sha=$phase4_identity_sha
phase4_split_identity "$trial_driver" 'trial driver'
phase4_split_identity "$correctness_verifier" 'correctness verifier'
correctness_verifier_sha=$phase4_identity_sha
phase4_split_identity "$identity_verifier" 'engine identity verifier'
identity_verifier_sha=$phase4_identity_sha
phase4_split_identity "$process_supervisor" 'process supervisor'
process_supervisor_sha=$phase4_identity_sha

[ "$(benchmark_campaign_field "$policy_path" 30 campaign_tool_sha256)" = \
    "$campaign_tool_sha" ] &&
  [ "$(benchmark_campaign_field "$policy_path" 31 correctness_verifier_sha256)" = \
    "$correctness_verifier_sha" ] &&
  [ "$(benchmark_campaign_field "$policy_path" 32 engine_identity_verifier_sha256)" = \
    "$identity_verifier_sha" ] &&
  [ "$(benchmark_campaign_field "$policy_path" 33 process_supervisor_sha256)" = \
    "$process_supervisor_sha" ] ||
  benchmark_campaign_fail 'externally expected verification tool identity changed'
[ "$(benchmark_campaign_field "$policy_path" 34 phase4_gate_executable_sha256)" = \
    "$gate_sha" ] ||
  benchmark_campaign_fail 'externally expected Phase 4 gate identity changed'
[ "$(benchmark_campaign_field "$policy_path" 35 phase4_fd_exec_helper_sha256)" = \
    "$helper_sha" ] ||
  benchmark_campaign_fail 'externally expected Phase 4 FD helper identity changed'
[ "$(benchmark_campaign_field "$policy_path" 38 physical_measurement_claim)" = none ] ||
  benchmark_campaign_fail 'Phase 4 policy makes a physical measurement claim'
[ "$(benchmark_campaign_field "$policy_path" 39 authority)" = "$authority" ] ||
  benchmark_campaign_fail 'Phase 4 policy authority changed'

phase4_scratch=$(mktemp -d /tmp/lunaflux-phase4-prefix-gate.XXXXXX) ||
  benchmark_campaign_fail 'could not create private Phase 4 scratch'
phase4_scratch=$(CDPATH= cd -- "$phase4_scratch" && pwd -P)
helper_pid=
phase4_cleanup() {
  if [ -n "$helper_pid" ]; then
    kill "$helper_pid" 2>/dev/null || true
    wait "$helper_pid" 2>/dev/null || true
  fi
  find "$phase4_scratch" -type d -exec chmod 700 {} \; 2>/dev/null || true
  rm -rf "$phase4_scratch"
}
trap phase4_cleanup EXIT HUP INT TERM
snapshot=$phase4_scratch/campaign
policy_copy=$phase4_scratch/phase4-policy.v1
gate_copy=$phase4_scratch/phase4-prefix-gate
helper_copy=$phase4_scratch/phase4-fd-exec
ready_fifo=$phase4_scratch/helper.ready
go_fifo=$phase4_scratch/helper.go
gate_stdout_path=$phase4_scratch/gate.stdout
gate_stderr_path=$phase4_scratch/gate.stderr

# The campaign is copied before inspection. A source mutation during the copy
# can only produce a snapshot that the exact campaign verifier rejects.
cp -Rp "$campaign_root" "$snapshot"
cp "$policy_path" "$policy_copy"
cp "$gate_path" "$gate_copy"
cp "$helper_path" "$helper_copy"
chmod 400 "$policy_copy"
chmod 500 "$gate_copy"
chmod 500 "$helper_copy"
benchmark_campaign_require_digest "$policy_copy" "$policy_sha" \
  'private Phase 4 policy'
benchmark_campaign_require_digest "$gate_copy" "$gate_sha" \
  'private Phase 4 gate executable'
benchmark_campaign_require_digest "$helper_copy" "$helper_sha" \
  'private Phase 4 FD execution helper'
mkfifo "$ready_fifo" "$go_fifo"
exec 3<>"$ready_fifo"
exec 4<>"$go_fifo"
: >"$gate_stdout_path"
: >"$gate_stderr_path"
exec 5>"$gate_stdout_path"
exec 6<"$gate_stdout_path"
exec 7>"$gate_stderr_path"
exec 8<"$gate_stderr_path"
rm "$gate_stdout_path" "$gate_stderr_path"

# Start the audited helper before any external verifier receives the scratch
# path. It opens and hashes the gate and policy, retains both descriptors, then
# blocks. On Linux it later uses execveat(AT_EMPTY_PATH); unsupported hosts fail
# closed. The test-only Darwin helper is separately hashed and has no authority.
"$helper_copy" "$gate_copy" "$gate_sha" "$policy_copy" "$policy_sha" \
  "$ready_fifo" "$go_fifo" phase4-prefix-gate --phase4-prefix-gate \
  "$snapshot" /dev/fd/8 "$policy_sha" "$gate_sha" \
  >&5 2>&7 3<&- 4<&- 5>&- 6<&- 7>&- 8<&- &
helper_pid=$!
helper_ready=
IFS= read -r helper_ready <&3 ||
  benchmark_campaign_fail 'Phase 4 FD helper failed before readiness'
[ "$helper_ready" = R ] ||
  benchmark_campaign_fail 'Phase 4 FD helper readiness changed'
rm "$ready_fifo" "$go_fifo"

phase4_verify_snapshot() {
  "$repo_root/scripts/verify-openai-comparison-campaign.sh" \
    "$snapshot" "$campaign_tool" "$trial_driver" "$correctness_verifier" \
    "$identity_verifier" "$process_supervisor" \
    3<&- 4<&- 5>&- 6<&- 7>&- 8<&- >/dev/null
}

phase4_verify_snapshot
gate_status=0
printf '%s' G >&4
wait "$helper_pid" || gate_status=$?
helper_pid=

# Re-authenticate the same private bytes after gate execution. Gate output is
# withheld until this succeeds. This software check itself claims no physical
# measurement, comparison authority, or release authority.
phase4_verify_snapshot
cat <&6
cat <&8 >&2
exit "$gate_status"
