#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
runner=$repo_root/scripts/run-openai-comparison-campaign.sh
verifier=$repo_root/scripts/verify-openai-comparison-campaign.sh

fail() {
  printf '%s\n' "LunaFlux external comparison campaign gate failed: $1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

. "$repo_root/scripts/validate-openai-comparison-invocation-evidence.sh"

assert_run_rejected() {
  label=$1
  recipe=$2
  output=$3
  recipe_sha=$(sha256_file "$recipe")
  if "$runner" "$recipe#sha256=$recipe_sha" "$output" \
    3</dev/null 4</dev/null 5</dev/null >/dev/null 2>&1; then
    fail "hostile fake-engine campaign passed: $label"
  fi
  [ ! -e "$output" ] || fail "failed campaign retained output: $label"
}

sh -n "$repo_root/scripts/benchmark-campaign-common.sh"
sh -n "$runner"
sh -n "$verifier"

fixture=$(mktemp -d /tmp/lunaflux-openai-campaign-gate.XXXXXX)
fixture=$(CDPATH= cd -- "$fixture" && pwd -P)
trap 'chmod -R u+w "$fixture" 2>/dev/null || true; rm -rf "$fixture"' EXIT HUP INT TERM

declaration=$fixture/campaign.declaration.json
printf '%s\n' '{"fixture_only":true,"authority":"none"}' > "$declaration"
declaration_sha=$(sha256_file "$declaration")

# These executables are intentionally fake and prove wiring only. Their exact
# digests are fixture identities and can never establish physical measurements.
campaign_tool=$fixture/fake-campaign-tool
cat > "$campaign_tool" <<'EOF'
#!/bin/sh
set -eu
for credential_fd in 3 4 5; do
  if (eval ": <&$credential_fd") 2>/dev/null; then
    exit 79
  fi
done
if [ "$1" = --preflight ]; then
  declaration_sha=$3
  cat <<EOT
schema=lunaflux-openai-responses-campaign-preflight.v1
declaration_sha256=$declaration_sha
engine_order=lunaflux,vllm,sglang
profile_order=latency,chat,long_prefill,decode_heavy,prefix_rich,prefix_cold,saturation,churn,mixed
trial_order=profile_then_ordinal_then_counterbalanced_engine
trial_count=81
credential_ingress=external.inherited-descriptor.v1
engine_0=lunaflux
engine_0_revision_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
engine_0_image_sha256=1111111111111111111111111111111111111111111111111111111111111111
engine_0_configuration_sha256=4444444444444444444444444444444444444444444444444444444444444444
engine_0_executable_sha256=7777777777777777777777777777777777777777777777777777777777777777
endpoint_0=127.0.0.1:8100
engine_1=vllm
engine_1_revision_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
engine_1_image_sha256=2222222222222222222222222222222222222222222222222222222222222222
engine_1_configuration_sha256=5555555555555555555555555555555555555555555555555555555555555555
engine_1_executable_sha256=8888888888888888888888888888888888888888888888888888888888888888
endpoint_1=127.0.0.1:8101
engine_2=sglang
engine_2_revision_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
engine_2_image_sha256=3333333333333333333333333333333333333333333333333333333333333333
engine_2_configuration_sha256=6666666666666666666666666666666666666666666666666666666666666666
engine_2_executable_sha256=9999999999999999999999999999999999999999999999999999999999999999
endpoint_2=127.0.0.1:8102
comparison_authority=none
EOT
  exit 0
fi
root=$1
[ "$(find "$root" -maxdepth 1 -type f -name 'trial-*.capture.json' | wc -l | tr -d ' ')" -eq 81 ] || exit 71
for capture in "$root"/trial-*.capture.json; do
  grep -F '"framing":"responses-sse-valid"' "$capture" >/dev/null || exit 72
done
cat <<'EOT'
schema=lunaflux-openai-responses-campaign-replay.v1
declaration_sha256=7777777777777777777777777777777777777777777777777777777777777777
raw_capture_set_sha256=8888888888888888888888888888888888888888888888888888888888888888
trial_count=81
observation_protocol=openai.responses.sse.v1
timestamp_authority=external_live_runner
network_authority=external_live_runner
credential_authority=external_live_runner
replay_authority=offline_admission_only
raw_artifact_mutation=none
comparison_authority=none
correctness_authority=none
replay_record_sha256=9999999999999999999999999999999999999999999999999999999999999999
EOT
EOF
chmod 755 "$campaign_tool"

driver=$fixture/fake-trial-driver
cat > "$driver" <<'EOF'
#!/bin/sh
set -eu
[ "$#" -eq 14 ] && [ "$1" = run ] || exit 80
index=$4
profile=$5
engine=$6
ordinal=$7
position=$8
capture=$9
credential_fd=${10}
revision=${11}
image=${12}
configuration=${13}
executable=${14}
case "$credential_fd" in 3|4|5) ;; *) exit 81 ;; esac
for candidate in 3 4 5; do
  if [ "$candidate" = "$credential_fd" ]; then
    (eval ": <&$candidate") 2>/dev/null || exit 84
  elif (eval ": <&$candidate") 2>/dev/null; then
    exit 85
  fi
done
printf '{"fixture_only":true,"trial_index":"%s","profile":"%s","engine":"%s","ordinal":"%s","order_position":"%s","revision_sha256":"%s","image_sha256":"%s","configuration_sha256":"%s","executable_sha256":"%s","framing":"responses-sse-valid"}\n' \
  "$index" "$profile" "$engine" "$ordinal" "$position" "$revision" "$image" \
  "$configuration" "$executable" > "$capture"
EOF
chmod 755 "$driver"

correctness=$fixture/fake-correctness-verifier
cat > "$correctness" <<'EOF'
#!/bin/sh
set -eu
for credential_fd in 3 4 5; do
  if (eval ": <&$credential_fd") 2>/dev/null; then
    exit 79
  fi
done
[ "$#" -eq 7 ] && [ "$1" = verify ] || exit 82
declaration_sha=$3
capture_sha=$5
index=$6
output=$7
cat > "$output" <<EOT
schema=lunaflux.external-correctness-observation.v1
declaration_sha256=$declaration_sha
trial_index=$index
capture_sha256=$capture_sha
reference_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
correctness_passed=1
EOT
EOF
chmod 755 "$correctness"

identity=$fixture/fake-engine-identity-verifier
cat > "$identity" <<'EOF'
#!/bin/sh
set -eu
for credential_fd in 3 4 5; do
  if (eval ": <&$credential_fd") 2>/dev/null; then
    exit 79
  fi
done
[ "$#" -eq 11 ] && [ "$1" = verify ] || exit 83
cat > "${11}" <<EOT
schema=lunaflux.live-engine-identity.v1
declaration_sha256=$3
trial_index=$4
engine=$5
endpoint=$6
revision_sha256=$7
image_sha256=$8
configuration_sha256=$9
executable_sha256=${10}
live_identity_verified=1
EOT
EOF
chmod 755 "$identity"

supervisor=$fixture/fake-process-supervisor
cat > "$supervisor" <<'EOF'
#!/bin/sh
set -eu
[ "${LUNAFLUX_HOSTILE_VERIFIER_SECRET+x}" != x ] || exit 78
[ "$1" = run ] || exit 84
timeout=$2
grace=$3
invocation=$4
invocation_sha=$5
stdout=$6
stderr=$7
receipt=$8
[ "$9" = -- ] || exit 85
shift 9
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
field() {
  value=$(sed -n "s/^$1=//p" "$invocation")
  [ -n "$value" ] && [ "$(grep -c "^$1=" "$invocation")" -eq 1 ] || exit 86
  printf '%s\n' "$value"
}
[ "$(sha256_file "$invocation")" = "$invocation_sha" ] || exit 87
[ "$(field schema)" = lunaflux.external-process-invocation.v1 ] || exit 88
[ "$(field timeout_seconds)" = "$timeout" ] || exit 89
[ "$(field grace_seconds)" = "$grace" ] || exit 90
[ "$(field argument_count)" = "$#" ] || exit 91
scope=$(field credential_scope)
for candidate in 3 4 5; do
  if [ "$scope" = "fd-$candidate" ]; then
    (eval ": <&$candidate") 2>/dev/null || exit 92
  elif (eval ": <&$candidate") 2>/dev/null; then
    exit 93
  fi
done
argument_file=$(mktemp /tmp/lunaflux-fixture-supervisor-argument.XXXXXX)
trap 'rm -f "$argument_file"' EXIT HUP INT TERM
index=0
for value in "$@"; do
  printf '%s' "$value" > "$argument_file"
  key=$(printf 'argument_%03d_sha256' "$index")
  [ "$(field "$key")" = "$(sha256_file "$argument_file")" ] || exit 94
  index=$((index + 1))
done
status=0
"$@" > "$stdout" 2> "$stderr" || status=$?
cat > "$receipt" <<EOT
schema=lunaflux.external-process-supervisor.v2
invocation_sha256=$invocation_sha
outcome=completed
exit_status=$status
timed_out=0
cancelled=0
process_group_empty=1
stdout_closed=1
stderr_closed=1
EOT
rm -f "$argument_file"
trap - EXIT HUP INT TERM
[ "$status" -eq 0 ]
EOF
chmod 755 "$supervisor"

write_recipe() {
  output=$1
  selected_driver=$2
  selected_correctness=$3
  selected_identity=$4
  selected_supervisor=$5
  cat > "$output" <<EOF
schema=lunaflux.openai-external-process-campaign.v1
declaration_source=$declaration
declaration_sha256=$declaration_sha
campaign_tool_source=$campaign_tool
campaign_tool_sha256=$(sha256_file "$campaign_tool")
trial_driver_source=$selected_driver
trial_driver_sha256=$(sha256_file "$selected_driver")
correctness_verifier_source=$selected_correctness
correctness_verifier_sha256=$(sha256_file "$selected_correctness")
engine_identity_verifier_source=$selected_identity
engine_identity_verifier_sha256=$(sha256_file "$selected_identity")
process_supervisor_source=$selected_supervisor
process_supervisor_sha256=$(sha256_file "$selected_supervisor")
source_identity_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
trial_timeout_seconds=30
tool_timeout_seconds=30
cancellation_grace_seconds=5
credential_fds=3,4,5
protocol=openai.responses.sse.v1
matrix=3-engines,9-profiles,3-trials
counterbalance=latin-square-v1
authority=capture-only-no-comparison
EOF
}

recipe=$fixture/campaign.recipe.v1
write_recipe "$recipe" "$driver" "$correctness" "$identity" "$supervisor"
recipe_sha=$(sha256_file "$recipe")
valid=$fixture/valid
"$runner" "$recipe#sha256=$recipe_sha" "$valid" \
  3</dev/null 4</dev/null 5</dev/null >/dev/null

campaign_argument=$campaign_tool#sha256=$(sha256_file "$campaign_tool")
driver_argument=$driver#sha256=$(sha256_file "$driver")
correctness_argument=$correctness#sha256=$(sha256_file "$correctness")
identity_argument=$identity#sha256=$(sha256_file "$identity")
supervisor_argument=$supervisor#sha256=$(sha256_file "$supervisor")
LUNAFLUX_HOSTILE_VERIFIER_SECRET=must-not-cross \
  "$verifier" "$valid" "$campaign_argument" "$driver_argument" \
  "$correctness_argument" "$identity_argument" "$supervisor_argument" \
  3</dev/null 4</dev/null 5</dev/null >/dev/null

[ "$(find "$valid/process" -type f | wc -l | tr -d ' ')" -eq 652 ] ||
  fail 'sealed process evidence lost invocation or receipt records'
for coordinate in \
  preflight:none \
  identity-0:none \
  trial-0:fd-3 \
  trial-1:fd-4 \
  trial-2:fd-5 \
  correctness-0:none \
  replay:none; do
  label=${coordinate%%:*}
  scope=${coordinate#*:}
  grep -F -x "credential_scope=$scope" \
    "$valid/process/$label.invocation.v1" >/dev/null 2>&1 ||
    fail "sealed invocation lost credential confinement: $label"
done

[ "$(wc -l < "$valid/execution-order.csv" | tr -d ' ')" -eq 81 ] ||
  fail 'counterbalanced execution order is incomplete'
[ "$(sed -n '1p' "$valid/execution-order.csv")" = 'latency,1,0,lunaflux,0' ] &&
  [ "$(sed -n '4p' "$valid/execution-order.csv")" = 'latency,2,0,vllm,4' ] &&
  [ "$(sed -n '7p' "$valid/execution-order.csv")" = 'latency,3,0,sglang,8' ] ||
  fail 'engine execution order is not counterbalanced'

if "$runner" "$recipe#sha256=$recipe_sha" "$valid" \
  3</dev/null 4</dev/null 5</dev/null >/dev/null 2>&1; then
  fail 'campaign runner overwrote a completed output'
fi

bad_identity=$fixture/bad-identity
sed 's/image_sha256=$8/image_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff/' \
  "$identity" > "$bad_identity"
chmod 755 "$bad_identity"
bad_identity_recipe=$fixture/bad-identity.recipe
write_recipe "$bad_identity_recipe" "$driver" "$correctness" "$bad_identity" "$supervisor"
assert_run_rejected fake-engine-image-substitution "$bad_identity_recipe" \
  "$fixture/bad-identity-output"

bad_executable_identity=$fixture/bad-executable-identity
sed 's/executable_sha256=${10}/executable_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff/' \
  "$identity" > "$bad_executable_identity"
chmod 755 "$bad_executable_identity"
bad_executable_recipe=$fixture/bad-executable.recipe
write_recipe "$bad_executable_recipe" "$driver" "$correctness" \
  "$bad_executable_identity" "$supervisor"
assert_run_rejected fake-engine-executable-substitution \
  "$bad_executable_recipe" "$fixture/bad-executable-output"

changing_identity=$fixture/changing-identity
cat > "$changing_identity" <<EOF
#!/bin/sh
set -eu
for credential_fd in 3 4 5; do
  if (eval ": <&\$credential_fd") 2>/dev/null; then
    exit 79
  fi
done
state=$fixture/changing-identity-state
count=0
[ ! -f "\$state" ] || count=\$(sed -n '1p' "\$state")
count=\$((count + 1))
printf '%s\n' "\$count" > "\$state"
image=\$8
if [ "\$count" -eq 2 ]; then
  image=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
fi
cat > "\${11}" <<EOT
schema=lunaflux.live-engine-identity.v1
declaration_sha256=\$3
trial_index=\$4
engine=\$5
endpoint=\$6
revision_sha256=\$7
image_sha256=\$image
configuration_sha256=\$9
executable_sha256=\${10}
live_identity_verified=1
EOT
EOF
chmod 755 "$changing_identity"
changing_identity_recipe=$fixture/changing-identity.recipe
write_recipe "$changing_identity_recipe" "$driver" "$correctness" \
  "$changing_identity" "$supervisor"
assert_run_rejected engine-identity-changed-during-trial \
  "$changing_identity_recipe" "$fixture/changing-identity-output"

bad_driver=$fixture/bad-framing-driver
sed 's/responses-sse-valid/responses-sse-invalid/' "$driver" > "$bad_driver"
chmod 755 "$bad_driver"
bad_driver_recipe=$fixture/bad-driver.recipe
write_recipe "$bad_driver_recipe" "$bad_driver" "$correctness" "$identity" "$supervisor"
assert_run_rejected malformed-captured-output "$bad_driver_recipe" \
  "$fixture/bad-driver-output"

bad_correctness=$fixture/bad-correctness
sed 's/correctness_passed=1/correctness_passed=0/' "$correctness" > "$bad_correctness"
chmod 755 "$bad_correctness"
bad_correctness_recipe=$fixture/bad-correctness.recipe
write_recipe "$bad_correctness_recipe" "$driver" "$bad_correctness" "$identity" "$supervisor"
assert_run_rejected failed-correctness "$bad_correctness_recipe" \
  "$fixture/bad-correctness-output"

timeout_supervisor=$fixture/timeout-supervisor
sed -e 's/outcome=completed/outcome=timed_out/' \
  -e 's/timed_out=0/timed_out=1/' "$supervisor" > "$timeout_supervisor"
chmod 755 "$timeout_supervisor"
timeout_recipe=$fixture/timeout.recipe
write_recipe "$timeout_recipe" "$driver" "$correctness" "$identity" "$timeout_supervisor"
assert_run_rejected timeout-cleanup "$timeout_recipe" "$fixture/timeout-output"

replay_supervisor=$fixture/replay-supervisor
sed 's/invocation_sha256=$invocation_sha/invocation_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/' \
  "$supervisor" > "$replay_supervisor"
chmod 755 "$replay_supervisor"
replay_supervisor_recipe=$fixture/replay-supervisor.recipe
write_recipe "$replay_supervisor_recipe" "$driver" "$correctness" "$identity" \
  "$replay_supervisor"
assert_run_rejected cleanup-receipt-replay "$replay_supervisor_recipe" \
  "$fixture/replay-supervisor-output"

noncanonical_timeout_recipe=$fixture/noncanonical-timeout.recipe
sed 's/trial_timeout_seconds=30/trial_timeout_seconds=030/' \
  "$recipe" > "$noncanonical_timeout_recipe"
assert_run_rejected noncanonical-timeout "$noncanonical_timeout_recipe" \
  "$fixture/noncanonical-timeout-output"

substituted=$fixture/substituted
cp -R "$valid" "$substituted"
chmod 755 "$substituted"
chmod 644 "$substituted/trial-000.capture.json"
printf '%s\n' substituted >> "$substituted/trial-000.capture.json"
chmod 444 "$substituted/trial-000.capture.json"
chmod 555 "$substituted"
if "$verifier" "$substituted" "$campaign_argument" "$driver_argument" \
  "$correctness_argument" \
  "$identity_argument" "$supervisor_argument" >/dev/null 2>&1; then
  fail 'verifier accepted a substituted raw capture'
fi

assert_recomputed_invocation_scope_rejected \
  "$fixture" "$valid" "$verifier" "$campaign_argument" "$driver_argument" \
  "$correctness_argument" "$identity_argument" "$supervisor_argument"

alternate_identity=$fixture/alternate-identity
cp "$identity" "$alternate_identity"
printf '%s\n' '# different unapproved verifier identity' >> "$alternate_identity"
chmod 755 "$alternate_identity"
alternate_identity_argument=$alternate_identity#sha256=$(sha256_file "$alternate_identity")
if "$verifier" "$valid" "$campaign_argument" "$driver_argument" \
  "$correctness_argument" \
  "$alternate_identity_argument" "$supervisor_argument" >/dev/null 2>&1; then
  fail 'verifier accepted an engine identity tool substitution'
fi

alternate_driver=$fixture/alternate-driver
cp "$driver" "$alternate_driver"
printf '%s\n' '# different unapproved trial driver identity' >> "$alternate_driver"
chmod 755 "$alternate_driver"
alternate_driver_argument=$alternate_driver#sha256=$(sha256_file "$alternate_driver")
if "$verifier" "$valid" "$campaign_argument" "$alternate_driver_argument" \
  "$correctness_argument" "$identity_argument" "$supervisor_argument" \
  >/dev/null 2>&1; then
  fail 'verifier accepted a trial driver substitution'
fi

grep -E '^engine_identity_verifier_sha256=[0-9a-f]{64}$' \
  "$valid/comparison-handoff.v1" >/dev/null 2>&1 ||
  fail 'handoff lost the exact live identity verifier digest'
for required in \
  'identity_artifact_count=81' \
  'process_cleanup_complete=1' \
  'comparison_admission=external-correctness-join-required' \
  'comparison_authority=none' \
  'physical_measurement_claim=none'; do
  grep -F -x "$required" "$valid/comparison-handoff.v1" >/dev/null 2>&1 ||
    fail "handoff lost authority boundary: $required"
done

printf '%s\n' 'LunaFlux external-process benchmark campaign hostile fake-engine gates passed.'
