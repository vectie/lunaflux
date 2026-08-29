#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/benchmark-campaign-common.sh"

scratch=$(mktemp -d /tmp/lunaflux-phase4-prefix-boundaries.XXXXXX) ||
  benchmark_campaign_fail 'could not create Phase 4 boundary scratch'
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
boundary_cleanup() {
  find "$scratch" -type d -exec chmod 700 {} \; 2>/dev/null || true
  rm -rf "$scratch"
}
trap boundary_cleanup EXIT HUP INT TERM

fake_repo=$scratch/repo
mkdir -p "$fake_repo/scripts" "$scratch/campaign" "$scratch/tools"
cp "$repo_root/scripts/benchmark-campaign-common.sh" "$fake_repo/scripts/"
cp "$repo_root/scripts/validate-phase4-prefix-benchmark.sh" \
  "$fake_repo/scripts/"

cat >"$fake_repo/scripts/verify-openai-comparison-campaign.sh" <<'EOF'
#!/bin/sh
set -eu
root=$1
[ "$(cat "$root/marker")" = sealed ] || exit 81
for private_name in helper.ready helper.go gate.stdout gate.stderr; do
  [ ! -e "$(dirname -- "$root")/$private_name" ] || exit 83
done
count=$(cat "${PHASE4_VERIFY_STATE:?}")
if [ "$count" -eq 0 ]; then
  chmod 600 "${PHASE4_ORIGINAL_ROOT:?}/marker"
  printf '%s\n' swapped >"$PHASE4_ORIGINAL_ROOT/marker"
  private_gate=$(dirname -- "$root")/phase4-prefix-gate
  case ${PHASE4_GATE_ATTACK:-replace} in
    replace)
      replacement=$(dirname -- "$root")/hostile-gate-replacement
      cat >"$replacement" <<'GATE'
#!/bin/sh
printf '%s\n' hostile-private-gate-executed
: >"${PHASE4_PRIVATE_EXECUTED:?}"
exit 79
GATE
      chmod 500 "$replacement"
      mv -f "$replacement" "$private_gate"
      ;;
    inplace)
      chmod 700 "$private_gate"
      printf '%s\n' '#!/bin/sh' 'exit 78' >"$private_gate"
      chmod 500 "$private_gate"
      ;;
    *) exit 82 ;;
  esac
fi
printf '%s\n' "$((count + 1))" >"$PHASE4_VERIFY_STATE"
EOF
chmod 500 "$fake_repo/scripts/verify-openai-comparison-campaign.sh"

printf '%s\n' sealed >"$scratch/campaign/marker"
chmod 444 "$scratch/campaign/marker"
chmod 555 "$scratch/campaign"
printf '%s\n' 0 >"$scratch/verify-state"
export PHASE4_ORIGINAL_ROOT=$scratch/campaign
export PHASE4_VERIFY_STATE=$scratch/verify-state
export PHASE4_PRIVATE_EXECUTED=$scratch/private-gate-executed

cat >"$scratch/phase4-gate" <<'EOF'
#!/bin/sh
set -eu
[ "$#" -eq 5 ]
[ "$1" = --phase4-prefix-gate ]
[ "$(cat "$2/marker")" = sealed ]
[ -f "$3" ]
[ "$(sed -n '1p' "$3")" = \
  schema=lunaflux.phase4-prefix-benchmark-policy.v1 ]
[ "${#4}" -eq 64 ]
[ "${#5}" -eq 64 ]
[ ! -e /dev/fd/2048 ]
printf '%s\n' \
  'schema=lunaflux.phase4-prefix-benchmark-gate.v1' \
  'phase4_prefix_gate=pass' \
  'physical_measurement_claim=none' \
  'comparison_authority=none' \
  'release_authority=none'
EOF
chmod 500 "$scratch/phase4-gate"

# Darwin cannot execute Mach-O by descriptor. This mock-only build exercises
# the same open/hash/re-hash/barrier logic while interpreting the fake shell
# gate through its pinned descriptor. It is never physical evidence authority.
cc -DPHASE4_TEST_SHELL_GATE -std=c11 -O2 -Wall -Wextra -Werror \
  "$repo_root/scripts/phase4-gate-fd-exec.c" -o "$scratch/phase4-fd-exec"
chmod 500 "$scratch/phase4-fd-exec"
cc -std=c11 -O2 -Wall -Wextra -Werror \
  "$repo_root/scripts/phase4-gate-fd-exec.c" \
  -o "$scratch/phase4-fd-exec-production"
chmod 500 "$scratch/phase4-fd-exec-production"

for tool in campaign-tool trial-driver correctness-verifier identity-verifier supervisor; do
  {
    printf '%s\n' '#!/bin/sh' 'exit 0'
    printf '# identity=%s\n' "$tool"
  } >"$scratch/tools/$tool"
  chmod 500 "$scratch/tools/$tool"
done

gate_sha=$(benchmark_campaign_sha256 "$scratch/phase4-gate")
helper_sha=$(benchmark_campaign_sha256 "$scratch/phase4-fd-exec")
campaign_sha=$(benchmark_campaign_sha256 "$scratch/tools/campaign-tool")
trial_driver_sha=$(benchmark_campaign_sha256 "$scratch/tools/trial-driver")
correctness_sha=$(benchmark_campaign_sha256 "$scratch/tools/correctness-verifier")
identity_sha=$(benchmark_campaign_sha256 "$scratch/tools/identity-verifier")
supervisor_sha=$(benchmark_campaign_sha256 "$scratch/tools/supervisor")

cat >"$scratch/phase4-policy.v1" <<EOF
schema=lunaflux.phase4-prefix-benchmark-policy.v1
declaration_sha256=1111111111111111111111111111111111111111111111111111111111111111
hardware_sha256=2222222222222222222222222222222222222222222222222222222222222222
driver_sha256=3333333333333333333333333333333333333333333333333333333333333333
toolkit_sha256=4444444444444444444444444444444444444444444444444444444444444444
model_sha256=5555555555555555555555555555555555555555555555555555555555555555
tokenizer_sha256=6666666666666666666666666666666666666666666666666666666666666666
corpus_sha256=7777777777777777777777777777777777777777777777777777777777777777
protocol_sha256=8888888888888888888888888888888888888888888888888888888888888888
randomization_sha256=9999999999999999999999999999999999999999999999999999999999999999
prefix_rich_workload_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
prefix_cold_workload_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
engine_0=lunaflux
engine_0_revision_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
engine_0_image_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
engine_0_configuration_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
engine_0_executable_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
engine_1=vllm
engine_1_revision_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
engine_1_image_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
engine_1_configuration_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
engine_1_executable_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
engine_2=sglang
engine_2_revision_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
engine_2_image_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
engine_2_configuration_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
engine_2_executable_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
capture_inventory_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
correctness_inventory_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
campaign_tool_sha256=$campaign_sha
correctness_verifier_sha256=$correctness_sha
engine_identity_verifier_sha256=$identity_sha
process_supervisor_sha256=$supervisor_sha
phase4_gate_executable_sha256=$gate_sha
phase4_fd_exec_helper_sha256=$helper_sha
cold_regression_budget_bps=1000
comparison_metric=aggregate_completed_requests_per_nanosecond
physical_measurement_claim=none
authority=phase4-measurement-gate-only
EOF
chmod 400 "$scratch/phase4-policy.v1"
policy_sha=$(benchmark_campaign_sha256 "$scratch/phase4-policy.v1")

if [ "$(uname -s)" != Linux ]; then
  mkfifo "$scratch/unsupported.ready" "$scratch/unsupported.go"
  exec 3<>"$scratch/unsupported.ready"
  exec 4<>"$scratch/unsupported.go"
  "$scratch/phase4-fd-exec-production" \
    "$scratch/phase4-gate" "$gate_sha" \
    "$scratch/phase4-policy.v1" "$policy_sha" \
    "$scratch/unsupported.ready" "$scratch/unsupported.go" \
    phase4-prefix-gate --phase4-prefix-gate \
    "$scratch/campaign" /dev/fd/8 "$policy_sha" "$gate_sha" \
    3<&- 4<&- >"$scratch/unsupported.stdout" \
    2>"$scratch/unsupported.stderr" &
  unsupported_pid=$!
  unsupported_ready=
  IFS= read -r unsupported_ready <&3
  [ "$unsupported_ready" = R ] ||
    benchmark_campaign_fail 'unsupported production helper readiness changed'
  rm "$scratch/unsupported.ready" "$scratch/unsupported.go"
  printf '%s' G >&4
  if wait "$unsupported_pid"; then
    benchmark_campaign_fail 'unsupported production helper executed a gate'
  fi
  grep -qx 'Phase 4 FD execution is supported only on Linux' \
    "$scratch/unsupported.stderr" ||
    benchmark_campaign_fail 'unsupported production helper did not fail closed'
  exec 3>&-
  exec 4>&-
fi

validator=$fake_repo/scripts/validate-phase4-prefix-benchmark.sh
valid_stdout=$scratch/valid.stdout
printf '%s\n' sentinel >"$scratch/high-fd-sentinel"
exec 2048<"$scratch/high-fd-sentinel"
"$validator" \
  "$scratch/campaign" \
  "$scratch/phase4-policy.v1#sha256=$policy_sha" \
  "$scratch/phase4-gate#sha256=$gate_sha" \
  "$scratch/phase4-fd-exec#sha256=$helper_sha" \
  "$scratch/tools/campaign-tool#sha256=$campaign_sha" \
  "$scratch/tools/trial-driver#sha256=$trial_driver_sha" \
  "$scratch/tools/correctness-verifier#sha256=$correctness_sha" \
  "$scratch/tools/identity-verifier#sha256=$identity_sha" \
  "$scratch/tools/supervisor#sha256=$supervisor_sha" \
  phase4-measurement-gate-only >"$valid_stdout"
exec 2048<&-
[ "$(cat "$scratch/campaign/marker")" = swapped ] ||
  benchmark_campaign_fail 'hostile source swap fixture did not execute'
[ "$(cat "$scratch/verify-state")" -eq 2 ] ||
  benchmark_campaign_fail 'private campaign snapshot was not verified twice'
[ ! -e "$PHASE4_PRIVATE_EXECUTED" ] ||
  benchmark_campaign_fail 'replaced private gate pathname was executed'
if grep -q 'hostile-private-gate-executed' "$valid_stdout"; then
  benchmark_campaign_fail 'private gate pathname replacement reached output'
fi
grep -qx 'physical_measurement_claim=none' "$valid_stdout" ||
  benchmark_campaign_fail 'measurement-only claim was not retained'
grep -qx 'comparison_authority=none' "$valid_stdout" ||
  benchmark_campaign_fail 'comparison authority unexpectedly appeared'
grep -qx 'release_authority=none' "$valid_stdout" ||
  benchmark_campaign_fail 'release authority unexpectedly appeared'

expect_rejection() {
  label=$1
  shift
  if "$@" >"$scratch/rejected.stdout" 2>"$scratch/rejected.stderr"; then
    benchmark_campaign_fail "$label was admitted"
  fi
}

chmod 700 "$scratch/campaign"
chmod 600 "$scratch/campaign/marker"
printf '%s\n' sealed >"$scratch/campaign/marker"
chmod 444 "$scratch/campaign/marker"
chmod 555 "$scratch/campaign"
printf '%s\n' 0 >"$scratch/verify-state"
export PHASE4_GATE_ATTACK=inplace
inplace_stdout=$scratch/inplace.stdout
"$validator" \
  "$scratch/campaign" \
  "$scratch/phase4-policy.v1#sha256=$policy_sha" \
  "$scratch/phase4-gate#sha256=$gate_sha" \
  "$scratch/phase4-fd-exec#sha256=$helper_sha" \
  "$scratch/tools/campaign-tool#sha256=$campaign_sha" \
  "$scratch/tools/trial-driver#sha256=$trial_driver_sha" \
  "$scratch/tools/correctness-verifier#sha256=$correctness_sha" \
  "$scratch/tools/identity-verifier#sha256=$identity_sha" \
  "$scratch/tools/supervisor#sha256=$supervisor_sha" \
  phase4-measurement-gate-only >"$inplace_stdout"
grep -qx 'phase4_prefix_gate=pass' "$inplace_stdout" ||
  benchmark_campaign_fail 'sealed gate did not survive in-place source mutation'
[ "$(cat "$scratch/verify-state")" -eq 2 ] ||
  benchmark_campaign_fail 'in-place attack snapshot was not verified twice'
export PHASE4_GATE_ATTACK=replace

expect_rejection 'substituted policy digest' \
  "$validator" "$scratch/campaign" \
  "$scratch/phase4-policy.v1#sha256=9999999999999999999999999999999999999999999999999999999999999999" \
  "$scratch/phase4-gate#sha256=$gate_sha" \
  "$scratch/phase4-fd-exec#sha256=$helper_sha" \
  "$scratch/tools/campaign-tool#sha256=$campaign_sha" \
  "$scratch/tools/trial-driver#sha256=$trial_driver_sha" \
  "$scratch/tools/correctness-verifier#sha256=$correctness_sha" \
  "$scratch/tools/identity-verifier#sha256=$identity_sha" \
  "$scratch/tools/supervisor#sha256=$supervisor_sha" \
  phase4-measurement-gate-only

cp "$scratch/phase4-gate" "$scratch/substituted-gate"
chmod 700 "$scratch/substituted-gate"
printf '%s\n' '# substituted' >>"$scratch/substituted-gate"
chmod 500 "$scratch/substituted-gate"
substituted_gate_sha=$(benchmark_campaign_sha256 "$scratch/substituted-gate")
expect_rejection 'substituted gate executable' \
  "$validator" "$scratch/campaign" \
  "$scratch/phase4-policy.v1#sha256=$policy_sha" \
  "$scratch/substituted-gate#sha256=$substituted_gate_sha" \
  "$scratch/phase4-fd-exec#sha256=$helper_sha" \
  "$scratch/tools/campaign-tool#sha256=$campaign_sha" \
  "$scratch/tools/trial-driver#sha256=$trial_driver_sha" \
  "$scratch/tools/correctness-verifier#sha256=$correctness_sha" \
  "$scratch/tools/identity-verifier#sha256=$identity_sha" \
  "$scratch/tools/supervisor#sha256=$supervisor_sha" \
  phase4-measurement-gate-only

cp "$scratch/tools/campaign-tool" "$scratch/tools/substituted-tool"
chmod 700 "$scratch/tools/substituted-tool"
printf '%s\n' '# substituted' >>"$scratch/tools/substituted-tool"
chmod 500 "$scratch/tools/substituted-tool"
substituted_tool_sha=$(benchmark_campaign_sha256 \
  "$scratch/tools/substituted-tool")
expect_rejection 'substituted campaign tool' \
  "$validator" "$scratch/campaign" \
  "$scratch/phase4-policy.v1#sha256=$policy_sha" \
  "$scratch/phase4-gate#sha256=$gate_sha" \
  "$scratch/phase4-fd-exec#sha256=$helper_sha" \
  "$scratch/tools/substituted-tool#sha256=$substituted_tool_sha" \
  "$scratch/tools/trial-driver#sha256=$trial_driver_sha" \
  "$scratch/tools/correctness-verifier#sha256=$correctness_sha" \
  "$scratch/tools/identity-verifier#sha256=$identity_sha" \
  "$scratch/tools/supervisor#sha256=$supervisor_sha" \
  phase4-measurement-gate-only

injected=$scratch/injected
expect_rejection 'shell metacharacter authority' \
  "$validator" "$scratch/campaign" \
  "$scratch/phase4-policy.v1#sha256=$policy_sha" \
  "$scratch/phase4-gate#sha256=$gate_sha" \
  "$scratch/phase4-fd-exec#sha256=$helper_sha" \
  "$scratch/tools/campaign-tool#sha256=$campaign_sha" \
  "$scratch/tools/trial-driver#sha256=$trial_driver_sha" \
  "$scratch/tools/correctness-verifier#sha256=$correctness_sha" \
  "$scratch/tools/identity-verifier#sha256=$identity_sha" \
  "$scratch/tools/supervisor#sha256=$supervisor_sha" \
  "phase4-measurement-gate-only;touch $injected"
[ ! -e "$injected" ] ||
  benchmark_campaign_fail 'shell metacharacters were evaluated'

printf '%s\n' \
  'Phase 4 private-snapshot and hostile-boundary checks passed.' \
  'physical_measurement_claim=none' \
  'comparison_authority=none' \
  'release_authority=none'
