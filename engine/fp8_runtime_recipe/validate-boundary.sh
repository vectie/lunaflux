#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
package="$root/engine/fp8_runtime_recipe"
interface="$package/pkg.generated.mbti"

moon info >/dev/null

if rg -n 'vectie/lunaflux/(device|runtime|service|scheduler|cmd)' \
  "$package/admit.mbt" "$package/canonical.mbt" "$package/types.mbt" \
  "$package/v2_admit.mbt" "$package/v2_canonical.mbt" \
  "$package/v2_types.mbt" "$package/v3_admit.mbt" \
  "$package/v3_canonical.mbt" "$package/v3_types.mbt" >/dev/null; then
  echo 'FP8 inert recipe imports device/runtime/service/scheduler/CLI authority' >&2
  exit 1
fi

for required in \
  'vectie/lunaflux/engine/fp8_startup_admission' \
  'vectie/lunaflux/kernels/fp8_launch_abi' \
  'vectie/lunaflux/kernels/numeric_capability_manifest' \
  'vectie/lunaflux/model/numeric_weight_file'; do
  rg -F -q "$required" "$package/moon.pkg" || {
    echo "FP8 inert recipe lost required authority: $required" >&2
    exit 1
  }
done

if rg -ni 'readiness|ready\(|execute|load_module|load_function|open_context' \
  "$interface" >/dev/null; then
  echo 'FP8 inert recipe public surface fabricates runtime authority' >&2
  exit 1
fi

rg -F -q \
  'pub fn admit(@fp8_startup_admission.Fp8StartupAdmission, @numeric_weight_file.NumericWeightFileInspection, ArrayView[@fp8_launch_abi.Fp8DynamicLaunchAbi], ArrayView[Fp8LaunchWeightRegionClaim], Fp8RuntimeRecipeLimits) -> Fp8RuntimeRecipe raise Fp8RuntimeRecipeError' \
  "$interface" || {
  echo 'FP8 exact runtime-recipe admission surface drifted' >&2
  exit 1
}

rg -F -q \
  'pub fn admit_v2(@fp8_startup_admission.Fp8StartupAdmissionV2, @numeric_weight_file.NumericWeightFileInspection, ArrayView[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi], ArrayView[Fp8LaunchWeightRegionClaimV2], Fp8RuntimeRecipeV2Limits) -> Fp8RuntimeRecipeV2 raise Fp8RuntimeRecipeV2Error' \
  "$interface" || {
  echo 'FP8 staged runtime-recipe v2 admission surface drifted' >&2
  exit 1
}

rg -F -q \
  'pub fn admit_reusable_paged_v3(Fp8RuntimeRecipeV2, @launch_contract.PagedKernelProfile, @device_layout.DeviceKvLayout, Fp8ReusablePagedRecipeV3Limits) -> Fp8ReusablePagedRecipeV3 raise Fp8ReusablePagedRecipeV3Error' \
  "$interface" || {
  echo 'FP8 reusable paged v3 recipe admission surface drifted' >&2
  exit 1
}

if rg -F -n 'Fp8DynamicLaunchAbi' "$interface" | \
  rg -F 'admit_v2' >/dev/null; then
  echo 'FP8 staged runtime recipe admits a legacy v1 launch type' >&2
  exit 1
fi

rg -F -q 'require_recipe_capacity(output, 8, maximum)' \
  "$package/canonical.mbt" || {
  echo 'FP8 canonical encoder lost incremental prefix bound' >&2
  exit 1
}
rg -F -q 'require_recipe_capacity(output, bytes.length(), maximum)' \
  "$package/canonical.mbt" || {
  echo 'FP8 canonical encoder lost incremental payload bound' >&2
  exit 1
}

rg -F -q 'require_v2_capacity(output, 8, maximum)' \
  "$package/v2_canonical.mbt" || {
  echo 'FP8 v2 canonical encoder lost incremental prefix bound' >&2
  exit 1
}
rg -F -q 'require_v2_capacity(output, bytes.length(), maximum)' \
  "$package/v2_canonical.mbt" || {
  echo 'FP8 v2 canonical encoder lost incremental payload bound' >&2
  exit 1
}
rg -F -q 'workspace_storage : @fp8_launch_abi.Fp8LaunchStorageIdentity' \
  "$package/v2_types.mbt" || {
  echo 'FP8 v2 recipe lost its singular workspace-arena identity' >&2
  exit 1
}
rg -F -q 'workspace_capacity_bytes : Int64' \
  "$package/v2_types.mbt" || {
  echo 'FP8 v2 recipe lost its singular workspace-arena capacity' >&2
  exit 1
}
rg -F -q 'launch.launch_source_version() != PagedNumericV4' \
  "$package/v2_admit.mbt" || {
  echo 'FP8 v2 recipe lost exact catalog-v4 raw authority validation' >&2
  exit 1
}
rg -F -q 'WeightScaleInput(weight, scale)' \
  "$package/v2_admit.mbt" || {
  echo 'FP8 v2 recipe lost paired weight-scale raw operand validation' >&2
  exit 1
}
for raw_authority in \
  'launch.launch_dimensions()' \
  'launch.raw_operands()' \
  'append_v2_raw_operand_role'; do
  rg -F -q "$raw_authority" "$package/v2_canonical.mbt" || {
    echo "FP8 v2 canonical lost raw authority: $raw_authority" >&2
    exit 1
  }
done

for reusable_invariant in \
  'fp8_reusable_token_capacity_contains(model_rows, model_tokens, max_tokens)' \
  'workspace_offset += workspace_bytes' \
  'launch.launch_source_version() != PagedNumericV4' \
  'append(reusable.workspace_offset().to_string())'; do
  rg -F -q "$reusable_invariant" \
    "$package/v3_admit.mbt" "$package/v3_canonical.mbt" || {
    echo "FP8 reusable paged v3 lost invariant: $reusable_invariant" >&2
    exit 1
  }
done

echo 'FP8 inert runtime-recipe boundary: pass'
