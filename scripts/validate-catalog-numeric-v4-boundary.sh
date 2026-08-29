#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

catalog_dir="kernels/catalog"

fail() {
  echo "catalog numeric-v4 boundary: $1" >&2
  exit 1
}

[ -d "$catalog_dir" ] || fail "missing catalog package"
production_files=$(rg --files "$catalog_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
[ -n "$production_files" ] || fail "catalog production source discovery is empty"

production_has() {
  rg -F -q "$1" "$catalog_dir" --glob '*.mbt' \
    --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'
}

for required in \
  'pub fn CatalogVersion::v4()' \
  'pub fn KernelSemanticVersion::v4()' \
  'pub fn CatalogEntry::new_paged_v4(' \
  'operation_execution_digest~ : @numeric_contract.OperationExecutionDigest' \
  'aot_entry_point~ : AotKernelEntryPoint' \
  'aot_entry_point.family() != family' \
  'operation_execution_digest: Some(operation_execution_digest)' \
  'aot_entry_point: Some(aot_entry_point)' \
  'pub fn KernelCatalog::resolve_paged_v4(' \
  'Some(require_operation_execution_digest(model_plan, operation.id()))' \
  'entry.operation_execution_digest == operation_execution_digest' \
  'existing.operation_execution_digest == entry.operation_execution_digest' \
  'pub fn KernelBinding::operation_execution_digest(' \
  'pub fn KernelBinding::aot_entry_point('; do
  production_has "$required" || \
    fail "missing exact v4 invariant: $required"
done

legacy_rejections=$(rg -F -c 'reject_legacy_i8_execution_contracts(model_plan)' \
  "$catalog_dir/catalog.mbt")
[ "$legacy_rejections" -eq 2 ] || \
  fail "legacy v1/v3 I8 rejection is not exact by resolver"

predicate_body=$(sed -n \
  '/^fn execution_contract_uses_i8_or_integer_accumulator(/,/^}/p' \
  "$catalog_dir/numeric_v4.mbt")
[ -n "$predicate_body" ] || fail "missing exhaustive legacy-I8 predicate"

dtype_accessors=$(sed -nE \
  's/^pub fn OperationExecutionContract::([a-z_]+)\(Self\) -> (ComputeDType|AccumulatorDType)$/\1/p' \
  model/numeric_contract/pkg.generated.mbti)
[ -n "$dtype_accessors" ] || fail "numeric dtype accessor discovery is empty"

predicate_is_exhaustive() {
  candidate=$1
  for accessor in $dtype_accessors; do
    printf '%s\n' "$candidate" | rg -F -q "execution.${accessor}()" || \
      return 1
  done
}

if predicate_is_exhaustive 'execution.activation_input() == i8'; then
  fail "legacy-I8 predicate positive control cannot detect omitted dtype fields"
fi
predicate_is_exhaustive "$predicate_body" || \
  fail "legacy-I8 predicate omits a current dtype accessor"

for accessor in $dtype_accessors; do
  if [ "$accessor" = accumulator ]; then
    required_comparison='execution.accumulator() == @numeric_contract.AccumulatorDType::i32()'
  else
    required_comparison="execution.${accessor}() == i8"
  fi
  printf '%s\n' "$predicate_body" | rg -F -q "$required_comparison" || \
    fail "legacy-I8 predicate does not reject $accessor exactly"
done

for required in \
  'CatalogSemanticVersionMismatch' \
  'AotEntryPointFamilyMismatch' \
  'InvalidVendorImplementationKind' \
  'UnsupportedNumericContract(@plan.OperationId)' \
  '(entry.operation_execution_digest is Some(_)) != expects_numeric_digest' \
  '(entry.aot_entry_point is Some(_)) != expects_numeric_digest'; do
  production_has "$required" || \
    fail "missing fail-closed v4 rule: $required"
done

forbidden_authority_pattern='kernels/(launch_contract|artifact|artifact_file)|engine/|scheduler/|internal/(cuda|nccl)|extern "c"'
for positive_control in \
  '"vectie/lunaflux/kernels/launch_contract"' \
  '"vectie/lunaflux/kernels/artifact"' \
  '"vectie/lunaflux/engine/device_step"' \
  '"vectie/lunaflux/scheduler/core"' \
  '"vectie/lunaflux/internal/cuda"' \
  'extern "c" fn hostile_catalog_authority'; do
  printf '%s\n' "$positive_control" | rg -q "$forbidden_authority_pattern" || \
    fail "forbidden-authority positive control is ineffective: $positive_control"
done

if rg -n "$forbidden_authority_pattern" \
  "$catalog_dir/moon.pkg" $production_files; then
  fail "catalog v4 acquired launch, artifact, scheduler, or backend authority"
fi

for required_api in \
  'pub fn CatalogVersion::v4() -> Self' \
  'pub fn KernelSemanticVersion::v4() -> Self' \
  'pub fn CatalogEntry::new_paged_v4' \
  'operation_execution_digest~ : @numeric_contract.OperationExecutionDigest' \
  'aot_entry_point~ : AotKernelEntryPoint' \
  'pub fn CatalogEntry::operation_execution_digest(Self) -> @numeric_contract.OperationExecutionDigest?' \
  'pub fn CatalogEntry::aot_entry_point(Self) -> AotKernelEntryPoint?' \
  'pub fn KernelCatalog::resolve_paged_v4' \
  'pub fn KernelBinding::operation_execution_digest(Self) -> @numeric_contract.OperationExecutionDigest?' \
  'pub fn KernelBinding::aot_entry_point(Self) -> AotKernelEntryPoint?'; do
  rg -F -q "$required_api" "$catalog_dir/pkg.generated.mbti" || \
    fail "generated catalog API drifted: $required_api"
done

for opaque_type in \
  AotArtifactDigest AotKernelFamilyId AotKernelFamily \
  AotKernelEntryPointId AotKernelEntryPoint CatalogVersion \
  KernelSemanticVersion DeviceTarget PagedKvLayoutContract \
  KernelExecutionSemantics CatalogEntry KernelCatalog KernelBinding \
  ResolvedKernelCatalog; do
  opaque_block=$(sed -n "/^pub struct ${opaque_type} {$/,/^} derive/p" \
    "$catalog_dir/pkg.generated.mbti")
  [ -n "$opaque_block" ] || \
    fail "validated catalog type is not an opaque record: $opaque_type"
  printf '%s\n' "$opaque_block" | rg -F -q '// private fields' || \
    fail "validated catalog fields are externally constructible: $opaque_type"
  if rg -q "^pub struct ${opaque_type}\\(" \
    "$catalog_dir/pkg.generated.mbti"; then
    fail "validated catalog tuple representation leaked: $opaque_type"
  fi
done

if rg -n '^pub fn (KernelBinding|ResolvedKernelCatalog)::new' \
  "$catalog_dir/pkg.generated.mbti"; then
  fail "resolved catalog evidence acquired an external constructor"
fi

for required_doc in \
  'Catalog v4 extends the paged-KV semantic contract' \
  '`OperationExecutionDigest` and catalog-owned `AotKernelEntryPoint`' \
  '`kernels/launch_contract` does not' \
  'select or replace v4 entry points'; do
  rg -F -q "$required_doc" "$catalog_dir/README.mbt.md" || \
    fail "catalog v4 documentation drifted: $required_doc"
done

for source_file in "$catalog_dir"/*.mbt; do
  case "$source_file" in
    *_test.mbt|*_wbtest.mbt) continue ;;
  esac
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  [ "$line_count" -lt 500 ] || \
    fail "$source_file exceeds the 499-line production budget"
done

echo "catalog numeric-v4 inert exact-selection boundary: ok"
