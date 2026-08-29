#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

numeric_dir="model/numeric_contract"
fixture_dir="model/plan_test_fixture"
device_source="device/numeric_capability.mbt"

fail() {
  echo "I8 numeric foundation boundary: $1" >&2
  exit 1
}

for path in "$numeric_dir" "$fixture_dir" "$device_source"; do
  [ -e "$path" ] || fail "missing $path"
done

for required in \
  'TensorStorageContract::symmetric_i8_weight_only_v1' \
  'OperationExecutionContract::symmetric_i8_weight_only_v1' \
  'symmetric_i8_weight_only_v1_accepts_code' \
  'shape.rank() == required_rank' \
  'target.shape.rank() == 1' \
  'claim_metadata_owner'; do
  rg -F -q "$required" "$numeric_dir" --glob '*.mbt' || \
    fail "numeric invariant is missing: $required"
done

fp8_policy=$(sed -n \
  '/^fn fp8_e4m3_architecture_allowed_v1(/,/^}/p' "$device_source")
i8_policy=$(sed -n \
  '/^fn symmetric_i8_weight_only_architecture_allowed_v1(/,/^}/p' \
  "$device_source")
for required in \
  '(major == 8 && minor == 9)' \
  '(major == 9 && minor == 0)' \
  '(major == 12 && minor == 0)'; do
  printf '%s\n' "$fp8_policy" | rg -F -q "$required" || \
    fail "FP8 exact architecture allowlist is missing: $required"
done
for required in \
  '(major == 8 && minor == 9)' \
  '(major == 9 && minor == 0)' \
  '(major == 12 && minor == 0)'; do
  printf '%s\n' "$i8_policy" | rg -F -q "$required" || \
    fail "I8 exact architecture allowlist is missing: $required"
done

for required in \
  'SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1' \
  'fn symmetric_i8_weight_only_architecture_allowed_v1' \
  '(major == 12 && minor == 0)' \
  'admit_symmetric_i8_weight_only_per_output_channel_v1' \
  'MissingBf16BoundarySupport' \
  'software target-admission allowlist' \
  'no artifact' \
  'physical support' \
  'readiness'; do
  rg -F -q "$required" "$device_source" || \
    fail "device invariant is missing: $required"
done

for required in \
  'synthetic_symmetric_i8_weight_only_numeric_schema_v1' \
  'validate_synthetic_i8_weight_only_selection_v1' \
  'QkvProjection | OutputProjection | GatedMlp | LanguageModelHead'; do
  rg -F -q "$required" "$fixture_dir" --glob '*.mbt' || \
    fail "closed fixture-selection invariant is missing: $required"
done

broad_architecture_pattern='(major|minor)[[:space:]]*>[=]?'
for positive_control in \
  'major > 8' 'major >= 8' 'minor > 0' 'minor >= 0'; do
  if ! printf '%s\n' "$positive_control" | rg -q "$broad_architecture_pattern"; then
    fail "architecture-ordering gate positive control is ineffective"
  fi
done

if rg -n "$broad_architecture_pattern" "$device_source"; then
  fail "device policy silently admits future architectures"
fi

if rg -n \
  'synthetic_symmetric_i8_weight_only_numeric_schema\(|admit_symmetric_i8_weight_only_per_output_channel\(' \
  "$fixture_dir" device scripts --glob '*.mbt' --glob '*.sh'; then
  fail "unversioned exact I8 v1 API remains reachable"
fi

if rg -n 'extern "c"|open_context|load_module|create_stream|\.run\(|\.execute\(' \
  "$numeric_dir" "$fixture_dir" "$device_source" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  fail "inert I8 foundation acquired physical execution authority"
fi

if rg -n 'symmetric_i8_weight_only|SymmetricI8WeightOnly' \
  kernels/catalog kernels/numeric_capability_manifest \
  engine/fp8_startup_admission model/device_materialize \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
  2>/dev/null; then
  fail "I8 foundation escaped into production catalog, manifest, loader, or materializer"
fi

if rg -n 'scheduler|kv/|worker_service|global.*precision|precision.*global' \
  "$numeric_dir" "$fixture_dir" "$device_source" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  fail "I8 foundation acquired scheduler, KV, service, or global policy"
fi

for directory in "$numeric_dir" "$fixture_dir" device; do
  for file in "$directory"/*.mbt; do
    case "$file" in
      *_test.mbt|*_wbtest.mbt) continue ;;
    esac
    lines=$(wc -l < "$file" | tr -d ' ')
    [ "$lines" -lt 500 ] || fail "$file exceeds the 499-line production budget"
  done
done

echo "symmetric-I8 numeric and inert device boundaries: ok"
