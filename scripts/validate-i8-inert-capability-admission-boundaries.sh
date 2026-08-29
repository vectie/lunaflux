#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

package_dir="kernels/i8_inert_capability_admission"

fail() {
  echo "I8 inert capability admission boundary: $1" >&2
  exit 1
}

is_production_source() {
  case "$1" in
    *_test.mbt|*_wbtest.mbt) return 1 ;;
    *.mbt) return 0 ;;
    *) return 1 ;;
  esac
}

extract_production_imports() {
  awk '
    /^[[:space:]]*import[[:space:]]*\{/ {
      in_import = 1
      import_count = 0
      next
    }
    in_import {
      if ($0 ~ /^[[:space:]]*"/) {
        path = $0
        sub(/^[^"]*"/, "", path)
        sub(/".*$/, "", path)
        imports[import_count] = path
        import_count += 1
      }
      if ($0 ~ /^[[:space:]]*}/) {
        test_only = $0 ~ /for[[:space:]]+"(test|wbtest)"/
        if (!test_only) {
          for (i = 0; i < import_count; i += 1) {
            print imports[i]
          }
        }
        delete imports
        import_count = 0
        in_import = 0
      }
    }
    END {
      if (in_import) {
        exit 2
      }
    }
  ' "$1"
}

is_allowed_production_import() {
  case "$1" in
    moonbitlang/core/encoding/utf8|\
    moonbitlang/x/crypto|\
    vectie/lunaflux/device|\
    vectie/lunaflux/kernels/catalog|\
    vectie/lunaflux/model/numeric_contract|\
    vectie/lunaflux/model/plan|\
    vectie/lunaflux/model/spec) return 0 ;;
    *) return 1 ;;
  esac
}

production_contains_fixed() {
  needle=$1
  for source_file in "$package_dir"/*.mbt; do
    if is_production_source "$source_file" && rg -F -q "$needle" "$source_file"; then
      return 0
    fi
  done
  return 1
}

production_matches() {
  expression=$1
  for source_file in "$package_dir"/*.mbt; do
    if is_production_source "$source_file" && rg -n -e "$expression" "$source_file"; then
      return 0
    fi
  done
  return 1
}

source_forbidden='extern[[:space:]]+"[cC]"|#external|module_bytes|function_symbol|artifact_module_index|DeviceContext|DeviceAllocation|KernelArtifactBundle|LaunchContract|DevicePlan|DeviceStep|Executor|Readiness|is_ready|pub fn .*\b(load|launch|execute|prepare|open)\b'
api_forbidden='KernelArtifact|LaunchContract|DevicePlan|DeviceStep|DeviceContext|DeviceAllocation|Executor|Readiness|ModuleInput|FunctionSymbol|->.*@plan\.ModelPlan([^A-Za-z0-9_]|$)|->.*@catalog\.ResolvedKernelCatalog([^A-Za-z0-9_]|$)|->.*@device\.DeviceNumericCapability([^A-Za-z0-9_]|$)'

# Positive controls keep every static predicate live instead of merely empty.
is_production_source "positive.mbt" || fail "production-source control failed"
if is_production_source "positive_test.mbt"; then
  fail "test-source exclusion control failed"
fi
is_allowed_production_import "vectie/lunaflux/device" || \
  fail "allowed-import control failed"
if is_allowed_production_import "evil/vectie/lunaflux/device"; then
  fail "exact-import rejection control failed"
fi
for sample in \
  '#external' \
  'KernelArtifactBundle' \
  'pub fn Bad::execute(self : Bad) -> Unit'; do
  printf '%s\n' "$sample" | rg -q -e "$source_forbidden" || \
    fail "production authority-pattern control failed: $sample"
done
for sample in \
  'pub fn Bad::plan(Self) -> @plan.ModelPlan' \
  'pub fn Bad::catalog(Self) -> (@catalog.ResolvedKernelCatalog, Int)' \
  'pub fn Bad::device(Self) -> @device.DeviceNumericCapability?' \
  'pub fn Bad::artifact(Self) -> KernelArtifactBundle'; do
  printf '%s\n' "$sample" | rg -q -e "$api_forbidden" || \
    fail "API leak-pattern control failed: $sample"
done
[ 499 -lt 500 ] || fail "line-budget positive control failed"
if [ 500 -lt 500 ]; then
  fail "line-budget rejection control failed"
fi

[ -d "$package_dir" ] || fail "package is missing"

for opaque_dependency in \
  'model/plan/pkg.generated.mbti:ModelPlan' \
  'model/plan/pkg.generated.mbti:ModelNumericBinding' \
  'kernels/catalog/pkg.generated.mbti:ResolvedKernelCatalog' \
  'kernels/catalog/pkg.generated.mbti:KernelBinding' \
  'kernels/catalog/pkg.generated.mbti:AotArtifactDigest' \
  'kernels/catalog/pkg.generated.mbti:AotKernelEntryPoint'; do
  interface_file=${opaque_dependency%%:*}
  type_name=${opaque_dependency#*:}
  rg -U -q "pub struct $type_name \\{\n  // private fields\n\\}" \
    "$interface_file" || \
    fail "trusted dependency type is no longer opaque: $type_name"
done

production_imports=$(extract_production_imports "$package_dir/moon.pkg") || \
  fail "moon.pkg import block is malformed"
production_import_count=$(printf '%s\n' "$production_imports" | sed '/^$/d' | wc -l | tr -d ' ')
[ "$production_import_count" -eq 7 ] || \
  fail "production import parser did not observe the exact seven-package surface"
printf '%s\n' "$production_imports" | rg -q -x 'vectie/lunaflux/device' || \
  fail "production import parser positive control failed"
if printf '%s\n' "$production_imports" | rg -q -x 'vectie/lunaflux/kv/device_layout'; then
  fail "test-only import escaped into the production import set"
fi
for imported in $production_imports; do
  is_allowed_production_import "$imported" || \
    fail "production import escaped the exact allowlist: $imported"
done

if production_matches "$source_forbidden"; then
  fail "production source exposes resource or execution authority"
fi

for required in \
  'pub struct I8InertKernelCapabilityAdmission {' \
  'pub fn admit_catalog_only_symmetric_i8_weight_only_v1(' \
  'device_numeric.feature() !=' \
  'SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1' \
  'resolved_v4.catalog_version() != @catalog.CatalogVersion::v4()' \
  'binding.semantic_version() != @catalog.KernelSemanticVersion::v4()' \
  'binding.operation_execution_digest() != Some(execution.digest())' \
  'binding.implementation() is ContentAddressedAot(family)' \
  'binding.aot_entry_point() is Some(entry_point)' \
  'entry_point.family() == family' \
  'kind is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead)' \
  'exact_i8_weight_storage(weight.storage())' \
  'scale_seen[scale_index] = true' \
  'scale.shape().dimension(0) == Some(output_channels)' \
  'max_workspace_bytes != resolved_v4.max_workspace_bytes()' \
  'max_workspace_alignment != resolved_v4.max_workspace_alignment()' \
  'i8_admission_v1_size_within(' \
  'additional > self.maximum - current' \
  'self.output.length() != self.expected' \
  'lunaflux.i8-inert-kernel-capability-admission.v1\x00'; do
  production_contains_fixed "$required" || \
    fail "mandatory production invariant is missing: $required"
done

admission_constructor_count=0
for source_file in "$package_dir"/*.mbt; do
  if is_production_source "$source_file"; then
    count=$(rg -n '^pub fn admit_catalog_only_symmetric_i8_weight_only_v1\(' \
      "$source_file" | wc -l | tr -d ' ')
    admission_constructor_count=$((admission_constructor_count + count))
  fi
done
[ "$admission_constructor_count" -eq 1 ] || \
  fail "admission constructor is not singular"

[ -f "$package_dir/pkg.generated.mbti" ] || fail "generated interface is missing"

for required_api in \
  'pub struct I8InertKernelCapabilityAdmission {' \
  'pub struct I8InertOperationEvidence {' \
  'pub struct I8InertWeightEvidence {' \
  'pub fn admit_catalog_only_symmetric_i8_weight_only_v1' \
  'pub fn I8InertKernelCapabilityAdmission::operations' \
  'pub fn I8InertOperationEvidence::entry_point' \
  'pub fn I8InertWeightEvidence::output_channel_count'; do
  rg -F -q "$required_api" "$package_dir/pkg.generated.mbti" || \
    fail "generated inert API drifted: $required_api"
done

if rg -n -e "$api_forbidden" "$package_dir/pkg.generated.mbti"; then
  fail "generated API leaks forbidden authority"
fi

for source_file in "$package_dir"/*.mbt "$package_dir"/*.mbt.md; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  [ "$line_count" -lt 500 ] || \
    fail "$source_file exceeds the strict 499-line budget"
done

echo "I8 catalog-only inert capability admission boundary: ok"
