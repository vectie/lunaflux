#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
producer=$repo_root/kernels/luna_bf16_kernel_producer
exporter=$repo_root/release/luna_bf16_candidate_export
command_dir=$repo_root/cmd/lunaflux_bf16_candidate_export

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

for root in "$producer" "$exporter" "$command_dir"; do
  find "$root" -type f -name '*.mbt' -print | while IFS= read -r source; do
    lines=$(wc -l <"$source" | tr -d ' ')
    [ "$lines" -lt 500 ] || fail "BF16 candidate export source exceeds 499 lines: $source"
  done
done

if rg -n \
  'internal/(cuda|process)|engine/|service/|scheduler/|ops/runtime_instance|nvcc|ptxas|nvrtc|cuModuleLoadData|popen[[:space:]]*\(|system[[:space:]]*\(|extern[[:space:]]+"[cC]"' \
  "$producer" "$exporter" "$command_dir" --glob '*.mbt' --glob 'moon.pkg'; then
  fail 'BF16 candidate export crossed compiler/device/process/runtime/service authority'
fi

producer_consumers=$(rg -l \
  'vectie/lunaflux/kernels/luna_bf16_kernel_producer' \
  "$repo_root" --glob 'moon.pkg' | sort)

producer_consumer_allowed() {
  case "$1" in
    "$repo_root/cmd/lunaflux_bf16_candidate_export/moon.pkg" | \
      "$repo_root/cmd/lunaflux_bf16_release_bind/moon.pkg" | \
      "$repo_root/kernels/luna_bf16_kernel_producer/moon.pkg" | \
      "$repo_root/release/kernel_root/moon.pkg" | \
      "$repo_root/release/luna_bf16_candidate_export/moon.pkg" | \
      "$repo_root/tests/approved_bf16_model_physical/moon.pkg") return 0 ;;
    *) return 1 ;;
  esac
}

while IFS= read -r consumer; do
  [ -z "$consumer" ] || producer_consumer_allowed "$consumer" ||
    fail "BF16 candidate producer acquired an unapproved package consumer: $consumer"
done <<EOF
$producer_consumers
EOF

release_bind_pkg=$repo_root/cmd/lunaflux_bf16_release_bind/moon.pkg
exact_producer_import='  "vectie/lunaflux/kernels/luna_bf16_kernel_producer",'
producer_import_allowed() {
  [ "$1" = "$exact_producer_import" ]
}

rg -F -x -q "$exact_producer_import" "$release_bind_pkg" ||
  fail 'closed release binder lost its exact producer import'

# Positive and hostile controls prove this is an exact package/import grant,
# not a prefix, suffix, near-name, or similar-import allowlist.
producer_consumer_allowed "$release_bind_pkg" ||
  fail 'exact closed release binder was not authorized'
if producer_consumer_allowed \
  "$repo_root/cmd/lunaflux_bf16_release_bind_shadow/moon.pkg"; then
  fail 'producer consumer allowlist accepted a near-name package'
fi
if producer_consumer_allowed \
  "$repo_root/cmd/lunaflux_bf16_release_bind/moon.pkg.injected"; then
  fail 'producer consumer allowlist accepted a suffixed package file'
fi
producer_import_allowed "$exact_producer_import" ||
  fail 'exact producer import was not authorized'
if producer_import_allowed \
  '  "vectie/lunaflux/kernels/luna_bf16_kernel_producer_shadow",'; then
  fail 'producer import allowlist accepted a near-name import'
fi
if producer_import_allowed \
  '  "vectie/lunaflux/kernels/luna_bf16_kernel_producer" @release_bind,'; then
  fail 'producer import allowlist accepted an aliased import'
fi

source_consumers=$(rg -l \
  '\.(candidates|candidate_source_bytes|candidate_recipe_bytes)\(' \
  "$repo_root" --glob '*.mbt' | sort)
source_consumer_allowed() {
  case "$1" in
    "$repo_root/cmd/lunaflux_bf16_release_bind/"*.mbt | \
      "$repo_root/kernels/luna_bf16_kernel_producer/"*.mbt | \
      "$repo_root/release/luna_bf16_candidate_export/"*.mbt | \
      "$repo_root/tests/approved_bf16_model_physical/"*.mbt) return 0 ;;
    *) return 1 ;;
  esac
}

while IFS= read -r consumer; do
  [ -z "$consumer" ] || source_consumer_allowed "$consumer" ||
    fail "candidate source/recipe reachability escaped offline authority: $consumer"
done <<EOF
$source_consumers
EOF

if source_consumer_allowed \
  "$repo_root/cmd/lunaflux_bf16_release_bind_shadow/compiled.mbt"; then
  fail 'candidate reconstruction allowlist accepted a near-name package'
fi

[ "$(rg -c 'lower_reusable_(pointwise|projection|paged_attention)_cuda_aot_candidate' "$producer/candidate_set.mbt")" -eq 3 ] ||
  fail 'candidate-set builder must call exactly the three product-owned reusable lowerers'
rg -q '^pub fn prepare_luna_bf16_candidate_set\(' "$producer/candidate_set.mbt" ||
  fail 'public authenticated-plan candidate-set builder is absent'
rg -q '^pub fn LunaBf16CandidateSet::candidates\(' "$producer/candidate_set_types.mbt" ||
  fail 'opaque candidate reconstruction view is absent'
rg -q 'validate_candidate_sequence\(model_plan, target, candidates\)' "$producer/candidate_set.mbt" ||
  fail 'production candidate sequence revalidation is absent'
rg -q 'sha256_bytes\(source\) != candidate_source_digest' "$producer/candidate_set.mbt" ||
  fail 'candidate source digest revalidation is absent'
rg -q 'sha256_bytes\(recipe\) != candidate_recipe_digest' "$producer/candidate_set.mbt" ||
  fail 'candidate recipe digest revalidation is absent'

rg -q 'create_mode=CreateNew' "$exporter/export.mbt" ||
  fail 'candidate exporter no-overwrite file writes are absent'
rg -q '@fs\.rename\(staging, absolute_new_output, replace=false\)' "$exporter/export.mbt" ||
  fail 'candidate exporter atomic no-replace publication is absent'
rg -q 'actual_parent != parent' "$exporter/path.mbt" ||
  fail 'candidate exporter canonical-parent check is absent'
rg -q 'ExportFailedAndCleanup' "$exporter/export.mbt" ||
  fail 'candidate exporter compound cleanup evidence is absent'

main_source=$command_dir/main.mbt
load_line=$(rg -n 'let plan = load_authenticated_single_row_plan' "$main_source" | cut -d: -f1)
prepare_line=$(rg -n 'let set = prepare_candidate_set' "$main_source" | cut -d: -f1)
export_line=$(rg -n 'luna_bf16_candidate_export\.export_new' "$main_source" | cut -d: -f1)
[ "$load_line" -lt "$prepare_line" ] && [ "$prepare_line" -lt "$export_line" ] ||
  fail 'CLI model admission, pure preparation, and publication ordering changed'
rg -q 'root\.close\(\)' "$main_source" || fail 'CLI approved-root close is absent'
rg -q 'compute_major=12' "$main_source" || fail 'CLI exact sm120 target is absent'
rg -q 'major=arguments\.compiler_major\.reinterpret_as_int\(\)' "$main_source" ||
  fail 'CLI exact compiler-major binding is absent'
rg -q 'minor=arguments\.compiler_minor\.reinterpret_as_int\(\)' "$main_source" ||
  fail 'CLI exact compiler-minor binding is absent'
rg -q 'patch=arguments\.compiler_patch\.reinterpret_as_int\(\)' "$main_source" ||
  fail 'CLI exact compiler-patch binding is absent'
rg -q 'value\.length\(\) > 6' "$command_dir/arguments.mbt" ||
  fail 'CLI compiler-version component bound is absent'
rg -q 'println\("physical_readiness=0"\)' "$main_source" ||
  fail 'CLI false physical-readiness evidence is absent'

rg -q 'candidate sequence rejects reordered and cross-target lowerings' \
  "$producer/candidate_set_wbtest.mbt" || fail 'candidate substitution test is absent'
rg -q 'rejects layout profile and capacity substitution' \
  "$producer/candidate_set_wbtest.mbt" || fail 'candidate input hostility test is absent'
rg -q 'rejects replacement and preserves first published bytes' \
  "$exporter/export_wbtest.mbt" || fail 'export no-overwrite test is absent'
rg -q 'rejects noncanonical unsafe and occupied staging paths' \
  "$exporter/export_wbtest.mbt" || fail 'export path hostility test is absent'

producer_mbti=$producer/pkg.generated.mbti
exporter_mbti=$exporter/pkg.generated.mbti
rg -q '^pub struct LunaBf16CandidateSet \{$' "$producer_mbti" ||
  fail 'candidate set is not abstract in generated MBTI'
rg -q '^pub fn LunaBf16CandidateSet::candidates\(' "$producer_mbti" ||
  fail 'candidate view is absent from generated MBTI'
candidate_shape=$(sed -n '/^pub struct LunaBf16CandidateSet {$/,/^}$/p' "$producer_mbti")
printf '%s\n' "$candidate_shape" | rg -q '^  // private fields$' ||
  fail 'candidate set does not retain private-field opacity in generated MBTI'
if printf '%s\n' "$candidate_shape" | rg -n '^  [^/]'; then
  fail 'candidate-set fields leaked through generated MBTI'
fi
if rg -n 'LunaBf16CandidatePayload' "$producer_mbti"; then
  fail 'candidate-set representation leaked through generated MBTI'
fi
rg -q '^pub struct LunaBf16CandidateExportEvidence \{$' "$exporter_mbti" ||
  fail 'export evidence is not abstract in generated MBTI'
export_shape=$(sed -n '/^pub struct LunaBf16CandidateExportEvidence {$/,/^}$/p' "$exporter_mbti")
printf '%s\n' "$export_shape" | rg -q '^  // private fields$' ||
  fail 'export evidence does not retain private-field opacity in generated MBTI'
if printf '%s\n' "$export_shape" | rg -n '^  [^/]'; then
  fail 'export evidence fields leaked through generated MBTI'
fi
if rg -n 'absolute_new_output|staging|source :|recipe :' "$exporter_mbti"; then
  fail 'filesystem or generated-byte storage leaked through export MBTI'
fi

printf '%s\n' 'Luna BF16 candidate export boundary gate passed'
