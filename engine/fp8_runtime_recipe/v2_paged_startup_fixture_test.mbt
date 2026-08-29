///|
fn runtime_recipe_v2_fixture(
  content? : String = "1111111111111111111111111111111111111111111111111111111111111111",
) -> RuntimeRecipeV2Fixture raise {
  let plan = recipe_v2_plan(content~)
  let resolved = recipe_v2_resolved(plan)
  let contracts = recipe_v2_contracts(plan, resolved)
  let exports : Array[@artifact.KernelEntryPointInput] = []
  for operation in 0..<6 {
    exports.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=recipe_v2_entry(operation),
        function_symbol="fp8_recipe_v2_\{operation}",
      ),
    )
  }
  let artifacts = try! @artifact.admit_paged_v4(
    contracts,
    [
      @artifact.KernelModuleInput::from_bytes(
        digest=recipe_module_digest(),
        module_bytes=b"fp8-recipe-module",
      ),
    ],
    exports,
    @artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=6,
      max_module_bytes=256L,
      max_total_module_bytes=256L,
      max_function_symbol_bytes=64,
    ),
  )
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let claims : Array[@numeric_capability_manifest.Fp8KernelManifestClaim] = []
  for operation in [1, 2, 3] {
    claims.push(
      @numeric_capability_manifest.Fp8KernelManifestClaim::new(
        profile~,
        operation=try! @plan.OperationId::new(operation),
        execution=plan.numeric_binding().operations()[operation]
          .execution()
          .digest(),
        entry_point=recipe_v2_entry(operation),
      ),
    )
  }
  let kernels = try! @numeric_capability_manifest.admit_paged_v4_fp8_e4m3_w8a8(
    plan,
    recipe_device(),
    resolved,
    contracts,
    artifacts,
    claims,
    @numeric_capability_manifest.Fp8KernelManifestLimits::new(
      max_entries=3,
      max_canonical_bytes=8192,
    ),
  )
  let limits = try! @fp8_startup_admission.Fp8StartupAdmissionLimits::new(
    max_canonical_bytes=8192,
  )
  {
    plan,
    legacy_startup: try! @fp8_startup_admission.admit(
      plan,
      recipe_device(),
      kernels,
      limits,
    ),
    startup: try! @fp8_startup_admission.admit_v2(
      plan,
      recipe_device(),
      kernels,
      limits,
    ),
    contracts,
  }
}
