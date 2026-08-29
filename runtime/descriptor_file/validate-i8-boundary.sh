#!/usr/bin/env bash
set -eu

runtime_dir="runtime/descriptor_file"
wire_dir="engine/worker_wire"
runtime_mbti="$runtime_dir/pkg.generated.mbti"
wire_mbti="$wire_dir/pkg.generated.mbti"

fail() {
  echo "$1" >&2
  exit 1
}

require_equal() {
  actual="$1"
  expected="$2"
  label="$3"
  if test "$actual" != "$expected"; then
    echo "$label differs from its exact allowlist" >&2
    echo "actual:" >&2
    echo "$actual" >&2
    exit 1
  fi
}

line_of() {
  file="$1"
  pattern="$2"
  matches="$(rg -n "$pattern" "$file" || true)"
  count="$(echo "$matches" | sed '/^$/d' | wc -l | tr -d ' ')"
  test "$count" = "1" || fail "ordering anchor is not unique: $pattern"
  echo "$matches" | sed 's/:.*//'
}

is_unique_before() {
  file="$1"
  first="$2"
  second="$3"
  first_line="$(line_of "$file" "$first")"
  second_line="$(line_of "$file" "$second")"
  test "$first_line" -lt "$second_line"
}

require_before() {
  is_unique_before "$1" "$2" "$3" ||
    fail "required preflight ordering is absent: $2 before $3"
}

test -f "$runtime_mbti"
test -f "$wire_mbti"

# The legacy loader remains present and separate.
rg -q '^pub fn load\(' "$runtime_dir/load.mbt"
rg -q '^pub fn load_i8_v2\(' "$runtime_dir/i8_load.mbt"
rg -q 'lunaflux\.runtime\.i8\.v2' "$runtime_dir/i8_schema.mbt"

# Pure scalar/lexical checks are exact positive controls, not name-only theater.
for anchor in \
  'ApprovedRelativeLocator::new(claims.model.config_locator)' \
  'claims.model.numeric_weights_locator' \
  'claims.kernels.manifest_locator' \
  'ModelConfigDigest::from_sha256' \
  'ContentDigest::from_sha256' \
  'NumericWeightArtifactDigest::from_sha256' \
  'ExecutionManifestDigest::from_sha256' \
  'WorkerBootstrapDigest::from_sha256' \
  'admit_ceilings(claims.ceilings)' \
  'admit_i8_device_capability(claims.execution)' \
  'claims.model.max_batch_rows != claims.worker.max_plan_rows'; do
  rg -F -q "$anchor" "$runtime_dir/i8_preflight.mbt" ||
    fail "missing I8 preflight anchor: $anchor"
done

require_before "$runtime_dir/i8_load.mbt" \
  'let claims = parse_i8_descriptor' 'preflight_i8_claims\(claims\)'
require_before "$runtime_dir/i8_load.mbt" \
  'preflight_i8_claims\(claims\)' 'let model = admit_i8_model'
require_before "$runtime_dir/i8_load.mbt" \
  'preflight_i8_claims\(claims\)' 'let weights = inspect_i8_weights'
# Positive control: the inverse must not satisfy the same ordering predicate.
if is_unique_before "$runtime_dir/i8_load.mbt" \
  'let model = admit_i8_model' 'preflight_i8_claims\(claims\)'; then
  fail "ordering gate positive control accepted an inverted flow"
fi

# Hostile evidence must exercise each release-critical early rejection.
test_file="$runtime_dir/i8_descriptor_file_test.mbt"
for anchor in \
  'I8 scalar and lexical failures precede hostile descendants' \
  'hostile-model-config' \
  'hostile-numeric-weights' \
  '../execution-i8-v4.json' \
  'deployment_approved_aot_only' \
  'new="\"max_batch_rows\":2"' \
  'AdmissionRejected(ExecutionManifest)' \
  'AdmissionRejected(BootstrapDigest)' \
  'AdmissionRejected(ModelPlan)' \
  'I8 schema rejects duplicate keys, byte overflow, and depth before descendants' \
  'DuplicateField' \
  'FileTooLarge(maximum=32L)' \
  'JsonDepthExceeded(maximum=1)'; do
  rg -F -q "$anchor" "$test_file" ||
    fail "missing hostile preflight test-body anchor: $anchor"
done

# Exact source and generated public allowlists prevent accidental authority API
# growth while permitting implementation-private preflight helpers.
expected_runtime_methods='bootstrap_digest
bootstrap_source
descriptor_digest
device_ordinal
execution
model_identity
startup_contract
target
weight_inspection'
source_runtime_methods="$(
  rg -o 'pub fn I8RuntimeDescriptorAdmission::[a-z_]+' \
    "$runtime_dir"/*.mbt | sed 's/.*:://' | sort
)"
mbti_runtime_methods="$(
  rg -o 'pub fn I8RuntimeDescriptorAdmission::[a-z_]+' "$runtime_mbti" |
    sed 's/.*:://' | sort
)"
require_equal "$source_runtime_methods" "$expected_runtime_methods" \
  "source I8 runtime methods"
require_equal "$mbti_runtime_methods" "$expected_runtime_methods" \
  "generated I8 runtime methods"

runtime_source_block="$(
  sed -n '/^pub struct I8RuntimeDescriptorAdmission {$/,/^}$/p' \
    "$runtime_dir/types.mbt"
)"
runtime_private_count="$(
  echo "$runtime_source_block" | rg -c '^  priv ' || true
)"
test "$runtime_private_count" = "7" ||
  fail "source I8 runtime admission is not exactly seven private fields"
if echo "$runtime_source_block" | rg -n '^  [a-z_]+[[:space:]]*:'; then
  fail "source I8 runtime admission exposes a public field"
fi
runtime_mbti_block="$(
  sed -n '/^pub struct I8RuntimeDescriptorAdmission {$/,/^}$/p' "$runtime_mbti"
)"
require_equal "$runtime_mbti_block" 'pub struct I8RuntimeDescriptorAdmission {
  // private fields
}' "generated I8 runtime opacity"

expected_wire_methods='i8_execution
i8_model_source'
source_wire_methods="$(
  rg -o 'pub fn EncodedBootstrapSource::i8_[a-z_]+' "$wire_dir"/*.mbt |
    sed 's/.*:://' | sort
)"
mbti_wire_methods="$(
  rg -o 'pub fn EncodedBootstrapSource::i8_[a-z_]+' "$wire_mbti" |
    sed 's/.*:://' | sort
)"
require_equal "$source_wire_methods" "$expected_wire_methods" \
  "source I8 wire accessors"
require_equal "$mbti_wire_methods" "$expected_wire_methods" \
  "generated I8 wire accessors"

for type_name in WorkerI8BootstrapExecution WorkerI8BootstrapModelSource; do
  block="$(rg -A1 "^pub struct $type_name \{" "$wire_mbti" | head -n 2)"
  require_equal "$block" "pub struct $type_name {
  // private fields" "generated $type_name opacity"
done

rg -q '^pub fn load_i8_v2\(.*\) -> I8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError$' \
  "$runtime_mbti"
rg -q '^pub fn decode_i8_bootstrap_source_v6\(FixedArray\[Byte\], Int\) -> EncodedBootstrapSource' \
  "$wire_mbti"
rg -q 'return decode_i8_bootstrap_source_v6' \
  "$wire_dir/bootstrap_source_decode.mbt"
for target_source in \
  "$wire_dir/bootstrap_source_i8_types.mbt" \
  "$runtime_dir/i8_admit_inputs.mbt"; do
  rg -F -q '(compute_major == 12 && compute_minor == 0)' "$target_source" ||
    rg -F -q \
      '(claims.compute_major == 12 && claims.compute_minor == 0)' \
      "$target_source" ||
    fail "sm120 is absent from the exact I8 target gate: $target_source"
done
if rg -n 'I8RuntimeDescriptorAdmission::.*(@device\.Context|@device\.Allocation|ApprovedRoot|Ready)' \
  "$runtime_mbti"; then
  fail "generated I8 runtime API leaks authority"
fi
if rg -n 'EncodedBootstrapSource::i8_.*-> (FixedArray\[Byte\]|@device\.|ApprovedRoot|Ready)' \
  "$wire_mbti"; then
  fail "generated I8 worker source API leaks storage or authority"
fi

echo "I8 preflight, descriptor, and worker-source boundaries: ok"
