#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
package=release/luna_tensor_core_physical_evidence

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    printf 'tensor-core physical evidence file exceeds budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

if rg -n \
  'moonbitlang/async|internal/(cuda|process|approved_fs)|vectie/lunaflux/(device|engine|runtime|service|cmd)/' \
  "$package/moon.pkg"; then
  printf '%s\n' 'tensor-core evidence gained active runtime authority' >&2
  exit 1
fi

if rg -n \
  'extern[[:space:]]+"c"|@fs\.|@process\.|@cuda\.|async[[:space:]]+fn|pub fn (open|spawn|compile|execute|load_module|bind_manifest|promote)\(' \
  "$package" --glob '*.mbt' --glob '!**/*_wbtest.mbt'; then
  printf '%s\n' 'tensor-core evidence source gained active authority' >&2
  exit 1
fi

for required in \
  'TENSOR_CORE_EVIDENCE_RESULT_LINES : Int = 67' \
  'lunaflux-lunatile-tensor-core-physical-campaign.v1' \
  'compiler_version", "13.1.115"' \
  'driver_record_sha256' \
  'cuobjdump_sass_instruction_count' \
  'nvdisasm_sass_instruction_count' \
  'instruction == "TCGEN05_MMA"' \
  'cases % 16384 != 0' \
  'local_bytes != 0' \
  'spill_store != 0' \
  'memcheck_error_summary_count' \
  'racecheck_error_summary_count' \
  'initcheck_error_summary_count' \
  'evidence_files_manifest_sha256' \
  'FILES.sha256\n' \
  'RESULT.txt\n' \
  'outer_seal != expected' \
  'manifest_bindable", "false"' \
  'promotion_authority", "absent"' \
  'pub fn lower_and_bind_externally_qualified_luna_tile_bf16_mma16x16x16('; do
  if ! rg -F -q "$required" "$package" --glob '*.mbt'; then
    printf 'tensor-core physical evidence invariant missing: %s\n' \
      "$required" >&2
    exit 1
  fi
done

for hostile in \
  'hostile tensor-core evidence substitution was admitted' \
  'mutated result passed the out-of-band outer seal' \
  'sealed evidence replayed onto a foreign candidate' \
  'manifest substitution was admitted' \
  'qualified evidence lost its serial/SIMT fallback identity'; do
  rg -F -q "$hostile" "$package" --glob '*_wbtest.mbt' || {
    printf 'tensor-core evidence hostile gate missing: %s\n' "$hostile" >&2
    exit 1
  }
done

for route in cmd device engine runtime service kernels/luna_kernel_bundle \
  kernels/luna_artifact_admission; do
  [ ! -d "$route" ] ||
    if rg -n \
      'vectie/lunaflux/release/luna_tensor_core_physical_evidence|QualifiedLunaTileBf16Mma16x16x16Artifact' \
      "$route" --glob '*.mbt' --glob 'moon.pkg' \
      --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
      printf 'unqualified production route consumes tensor-core evidence: %s\n' \
        "$route" >&2
      exit 1
    fi
done

rg -F -q \
  'if policy.tensor_core_policy == RequireExternallyQualifiedTensorCore {' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -F -q 'raise InvalidProgram(TensorCoreQualification)' \
  kernels/luna_tile_ir/parallel_specialize.mbt
rg -F -q 'caller-selected tensor-core qualification was accepted' \
  kernels/luna_tile_ir/parallel_test.mbt

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface=$package/pkg.generated.mbti
rg -U -q \
  'pub struct AdmittedLunaTensorCorePhysicalEvidence \{\n  // private fields\n\}' \
  "$interface"
rg -U -q \
  'pub struct QualifiedLunaTileBf16Mma16x16x16Artifact \{\n  // private fields\n\}' \
  "$interface"
rg -F -q \
  'pub fn admit_luna_tile_tensor_core_physical_evidence(' "$interface"
rg -F -q \
  'pub fn lower_and_bind_externally_qualified_luna_tile_bf16_mma16x16x16(' \
  "$interface"

if rg -n \
  '^pub fn (AdmittedLunaTensorCorePhysicalEvidence|QualifiedLunaTileBf16Mma16x16x16Artifact)::(new|make|create|open|load|execute|promote)' \
  "$interface"; then
  printf '%s\n' 'tensor-core evidence gained a fabricator or runtime method' >&2
  exit 1
fi

printf '%s\n' \
  'LunaTile sealed tensor-core qualification binder remains evidence-only: PASS'
