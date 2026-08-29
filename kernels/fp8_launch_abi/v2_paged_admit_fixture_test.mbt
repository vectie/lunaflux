///|
fn v2_paged_kernels(
  plan : @plan.ModelPlan,
  catalog : @catalog.ResolvedKernelCatalog,
  launches : @launch_contract.PagedV4LaunchContractSet,
) -> @numeric_capability_manifest.AdmittedFp8KernelArtifacts {
  let artifacts = try! @artifact.admit_paged_v4(
    launches,
    [
      @artifact.KernelModuleInput::new(
        digest=fixture_module_digest(),
        module_bytes=fixture_module_bytes(),
      ),
    ],
    [
      @artifact.KernelEntryPointInput::new(
        entry_point=fixture_entry(1, 10),
        function_symbol="embedding_bf16_v4",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=fixture_entry(2, 11),
        function_symbol="fp8_staged_v2",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=fixture_entry(3, 12),
        function_symbol="rotary_bf16_v4",
      ),
      @artifact.KernelEntryPointInput::new(
        entry_point=fixture_entry(4, 13),
        function_symbol="attention_bf16_v4",
      ),
    ],
    @artifact.KernelArtifactLimits::new(
      max_modules=1,
      max_entry_points=4,
      max_module_bytes=1024L,
      max_total_module_bytes=1024L,
      max_function_symbol_bytes=64,
    ),
  )
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=8,
    compute_minor=9,
    total_memory_bytes=16L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let numeric = try! @device.admit_finite_fp8_e4m3_w8a8(observed)
  let claim = @numeric_capability_manifest.Fp8KernelManifestClaim::new(
    profile=v2_paged_profile().id(),
    operation=try! @plan.OperationId::new(1),
    execution=plan.numeric_binding().operations()[1].execution().digest(),
    entry_point=fixture_entry(2, 11),
  )
  try! @numeric_capability_manifest.admit_paged_v4_fp8_e4m3_w8a8(
    plan,
    numeric,
    catalog,
    launches,
    artifacts,
    [claim],
    @numeric_capability_manifest.Fp8KernelManifestLimits::new(
      max_entries=4,
      max_canonical_bytes=4096,
    ),
  )
}

///|
fn v2_paged_fixture(gated : Bool) -> Fp8V2PagedFixture {
  let plan = v2_paged_plan(gated)
  let catalog = v2_paged_catalog(plan)
  let launches = try! v2_paged_launches(plan, catalog)
  let scales = if gated {
    [
      fixture_region(0L, 4L, 16L),
      fixture_region(16L, 4L, 16L),
      fixture_region(32L, 4L, 16L),
    ]
  } else {
    [fixture_region(0L, 4L, 16L)]
  }
  {
    plan,
    launches,
    kernels: v2_paged_kernels(plan, catalog, launches),
    weight_scales: scales,
    workspace: fixture_region(
      16L,
      if gated {
        8L
      } else {
        4L
      },
      4L,
      workspace=true,
    ),
    shape: try! @fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
      profile=v2_paged_profile().id(),
      rows=[
        @fp8_launch_abi.Fp8PagedRowShape::new(
          query_tokens=2,
          context_tokens=2,
          page_table_entries=1,
        ),
      ],
    ),
  }
}

///|
fn admit_v2_paged(
  fixture : Fp8V2PagedFixture,
  launches? : @launch_contract.PagedV4LaunchContractSet,
  workspace? : @fp8_launch_abi.Fp8LaunchStorageRegion,
  shape? : @fp8_launch_abi.Fp8ExactExecutionShape,
  weight_scales? : Array[@fp8_launch_abi.Fp8LaunchStorageRegion],
) -> @fp8_launch_abi.Fp8StagedDynamicLaunchAbi raise @fp8_launch_abi.Fp8LaunchAbiError {
  @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
    fixture.plan,
    fixture.kernels,
    PagedV4(launches.unwrap_or(fixture.launches)),
    operation=try! @plan.OperationId::new(1),
    shape=shape.unwrap_or(fixture.shape),
    weight_scale_regions=weight_scales.unwrap_or(fixture.weight_scales),
    workspace=workspace.unwrap_or(fixture.workspace),
  )
}
