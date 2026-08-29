#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

for package in runtime/instance_policy_file tokenizer/json_file runtime/instance_admission; do
  [ -f "$package/moon.pkg" ] || fail "missing instance-admission package: $package"
done

if rg -n 'runtime/(descriptor_file|instance_admission)|config/|scheduler/|service/|engine/' \
  runtime/instance_policy_file/moon.pkg; then
  fail 'policy snapshot/schema package gained runtime activation responsibility'
fi

if rg -n 'approved_fs|async/fs|core/json|json_guard|crypto' \
  runtime/instance_admission/moon.pkg runtime/instance_admission/*.mbt; then
  fail 'pure instance join gained filesystem or parser authority'
fi

if rg -n '@approved_fs\.(ApprovedRoot|ApprovedFile)|ApprovedRoot|ApprovedFile' \
  runtime/instance_admission --glob '*.mbt'; then
  fail 'instance admission retains approved filesystem authority'
fi

if ! rg -q 'lunaflux\.instance-policy\.v1' runtime/instance_policy_file/schema.mbt ||
  ! rg -U -q 'require_lower_sha256\(\s*root,\s*"runtime_descriptor_sha256"' \
    runtime/instance_policy_file/schema.mbt ||
  ! rg -U -q 'require_lower_sha256\(\s*object,\s*"sha256"' \
    runtime/instance_policy_file/schema.mbt ||
  ! rg -U -q 'require_relative_locator\(\s*object,\s*"locator"' \
    runtime/instance_policy_file/schema.mbt; then
  fail 'strict policy identity or tokenizer locator gate disappeared'
fi

if ! rg -q 'finish_policy_file' runtime/instance_policy_file/file_owner.mbt ||
  ! rg -q 'finish_tokenizer_file' tokenizer/json_file/load.mbt ||
  ! rg -q 'matches_source' tokenizer/json_file/types.mbt; then
  fail 'close-before-publication or tokenizer substitution gate disappeared'
fi

if ! rg -q '@runtime_resolved\.resolve' runtime/instance_admission/join.mbt ||
  ! rg -q 'require_preparation_lanes' runtime/instance_admission/join.mbt ||
  ! rg -q 'configured_lanes != resolved_slots' runtime/instance_admission/coherence.mbt ||
  ! rg -q 'plan\.required_capabilities()' runtime/instance_admission/join.mbt ||
  ! rg -q 'require_fits_transport_wait' runtime/instance_admission/join.mbt ||
  ! rg -q 'require_fits' runtime/instance_admission/join.mbt; then
  fail 'instance capacity, capability, transport, or preparation join disappeared'
fi

if ! rg -q '@online_session\.admitted_maximum_transport_wait_millis' \
  runtime/instance_admission/join.mbt ||
  ! rg -q 'admitted_maximum_transport_wait_millis\(self\.require_ready_service\(\)\.inference\)' \
    service/online_session/coordinator_prepare.mbt; then
  fail 'preparation owner is no longer the single maximum transport-wait rule source'
fi

policy_identity_calls=$(rg -n '\.require_absolute_identity\(' \
  runtime/instance_policy_file --glob '*.mbt' --glob '!*_test.mbt' \
  --glob '!*_wbtest.mbt' || true)
tokenizer_identity_calls=$(rg -n '\.require_absolute_identity\(' \
  tokenizer/json_file --glob '*.mbt' --glob '!*_test.mbt' \
  --glob '!*_wbtest.mbt' || true)
if [ "$(printf '%s\n' "$policy_identity_calls" | sed '/^$/d' | wc -l | tr -d ' ')" -ne 1 ] ||
  [ "$(printf '%s\n' "$tokenizer_identity_calls" | sed '/^$/d' | wc -l | tr -d ' ')" -ne 1 ]; then
  fail 'policy or tokenizer root identity binding is not singular and explicit'
fi

if ! rg -q 'plan_frame_byte_capacity' runtime/instance_admission/coherence.mbt ||
  ! rg -q 'completion_frame_byte_capacity' runtime/instance_admission/coherence.mbt ||
  ! rg -q 'bootstrap_source_capacity' runtime/instance_admission/coherence.mbt; then
  fail 'worker process exact frame-capacity gate disappeared'
fi

if rg -n '"schema_version", "model", "kernels", "execution"' \
  runtime/instance_policy_file tokenizer/json_file runtime/instance_admission; then
  fail 'instance policy duplicated or extended runtime descriptor v1 schema'
fi

while IFS= read -r source_file; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -ge 500 ]; then
    printf '%s: %s lines\n' "$source_file" "$line_count" >&2
    failed=1
  fi
done <<EOF
$(rg --files runtime/instance_policy_file tokenizer/json_file runtime/instance_admission \
  --glob '*.mbt' | sort)
EOF

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'instance policy, tokenizer file, and pure admission boundaries are valid'
