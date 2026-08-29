///|
fn bootstrap_limits(max_steps? : Int = 32) -> DeviceBootstrapLimits raise {
  DeviceBootstrapLimits::new(
    max_modules=4,
    max_entry_points=16,
    max_steps~,
    max_operands_per_step=16,
    max_symbol_bytes=64,
    max_canonical_bytes=16384,
  )
}

///|
fn bootstrap_inference_limits(
  max_context_tokens? : Int = 8,
  max_token_id? : Int = 7,
) -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens~,
    max_token_id~,
    max_stop_token_ids=2,
    max_stop_strings=1,
    max_stop_string_bytes=8,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=100UL,
    max_top_k=8,
    max_temperature=4.0,
  )
}

///|
fn bootstrap_worker_limits(
  max_prefill_rows? : Int = 2,
  max_decode_rows? : Int = 2,
  max_plan_rows? : Int = 2,
  max_plan_tokens? : Int = 4,
  max_plan_pages? : Int = 4,
  max_sequence_tokens? : Int = 8,
  max_token_id? : Int = 7,
) -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows~,
    max_decode_rows~,
    max_plan_rows~,
    max_plan_tokens~,
    max_plan_pages~,
    max_capabilities=64,
    max_completion_slots=32,
    max_sequence_tokens~,
    max_token_id~,
  )
}

///|
fn bootstrap_fixture(
  digest_head? : Char = '0',
  entry_bias? : Int = 0,
  max_sequence_tokens? : Int = 8,
  max_token_id? : Int = 7,
) -> (PagedKvExecutionBlueprint, @artifact.KernelArtifactBundle) raise {
  let metadata = blueprint_metadata(digest_head~)
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(
    plan,
    static_plan,
    memory,
    layout,
    target,
    entry_bias~,
  )
  let artifacts = blueprint_artifacts(contracts)
  let blueprint = admit_paged_graph_execution_blueprint(
    plan,
    DeviceStepLimits::new(
      max_rows=2,
      max_tokens=4,
      max_pages=4,
      max_sequence_tokens~,
      max_token_id~,
    ),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    artifacts,
  )
  (blueprint, artifacts)
}

///|
fn bootstrap_i8_digest_fixture() -> (
  @plan.ModelPlan,
  @device.DeviceNumericCapability,
  @i8_admission.I8InertKernelCapabilityAdmissionDigest,
  @catalog.DeviceTarget,
) raise {
  let base = blueprint_model_plan(blueprint_metadata())
  let selected = [@plan.OperationId::new(2)]
  let numeric_schema = @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
    base.operations(),
    selected,
  )
  let plan = @plan.ModelPlan::new_with_kv_execution(
    content_digest=base.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=base.shape_constraints(),
    kv_cache_geometry=base.kv_cache_geometry(),
    workspace_bounds=base.workspace_bounds(),
    operations=base.operations(),
    final_output=base.final_output(),
    kv_execution=base.kv_execution(),
    numeric_schema~,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let layout = blueprint_layout(plan)
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    let kv_layout = if operation.persistent_state() is KvCacheReadWrite(_) {
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=layout.geometry().layout(),
          stable_version=layout.stable_layout_version(),
          tokens_per_page=layout.tokens_per_page(),
        ),
      )
    } else {
      None
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=@catalog.KernelExecutionSemantics::paged_v3(
          kv_layer=operation.kv_layer(),
          runtime_inputs=operation.runtime_inputs(),
          persistent_state=operation.persistent_state(),
          kv_layout~,
        ),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=@catalog.AotKernelEntryPoint::new(
          family=@catalog.AotKernelFamily::new(
            module_digest=digest,
            family_id=@catalog.AotKernelFamilyId::from_stable_int(index + 1),
          ),
          entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
            index + 1,
          ),
        ),
        compute_major=8,
        compute_minor=9,
        implementation=ContentAddressedAot(
          @catalog.AotKernelFamily::new(
            module_digest=digest,
            family_id=@catalog.AotKernelFamilyId::from_stable_int(index + 1),
          ),
        ),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() > 0L {
          16L
        } else {
          1L
        },
      ),
    )
  }
  let resolved = @catalog.KernelCatalog::new(
    version=@catalog.CatalogVersion::v4(),
    entries~,
  ).resolve_paged_v4(plan, layout, target)
  let observed = @device.DeviceCapability::from_observed_facts(
    compute_major=8,
    compute_minor=9,
    total_memory_bytes=8L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let device_numeric = @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    observed,
  )
  let admission = @i8_admission.admit_catalog_only_symmetric_i8_weight_only_v1(
    plan,
    device_numeric,
    resolved,
    @i8_admission.I8InertKernelCapabilityAdmissionLimits::new(
      max_operations=16,
      max_weights=32,
      max_canonical_bytes=16384,
    ),
  )
  (plan, device_numeric, admission.digest(), target)
}

///|
test "I8 bootstrap-v3 startup method binds exact digest and runtime envelope" {
  let (plan, device_numeric, i8_digest, target) = bootstrap_i8_digest_fixture()
  let manifest : I8DeviceWorkerBootstrapManifestV3 = {
    digest: @worker_wire.WorkerBootstrapDigest::from_sha256(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    identity: plan.identity(),
    numeric_schema_digest: plan.numeric_binding().digest(),
    i8_admission_digest: i8_digest,
    device_numeric_digest: device_numeric.digest(),
    device_ordinal: 0,
    target,
    catalog_version: @catalog.CatalogVersion::v4(),
    limits: blueprint_limits(),
    module_count: 1,
    entry_point_count: plan.operations().length(),
    step_count: plan.operations().length(),
  }
  let source = @worker_wire.WorkerBootstrapSourceDigest::from_sha256(
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  )
  let startup = manifest.worker_startup_contract(
    @worker_protocol.ModelPlanGeneration::first(),
    bootstrap_source_digest=source,
    bootstrap_worker_limits(),
    bootstrap_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
  assert_eq(startup.identity(), manifest.identity())
  assert_eq(startup.bootstrap_digest(), manifest.digest())
  assert_eq(startup.bootstrap_source_digest(), source)
  try
    manifest.worker_startup_contract(
      @worker_protocol.ModelPlanGeneration::first(),
      bootstrap_source_digest=source,
      bootstrap_worker_limits(max_sequence_tokens=9),
      bootstrap_inference_limits(),
      has_predecessor=false,
      predecessor_value=0UL,
    )
  catch {
    InvalidBootstrap(RuntimeEnvelope) => ()
    _ => fail("unexpected I8 startup-envelope rejection")
  } noraise {
    _ => fail("oversized I8 worker envelope was admitted")
  }
}

///|
test "device worker bootstrap digest is canonical and complete" {
  let (blueprint, artifacts) = bootstrap_fixture()
  let first = admit_device_worker_bootstrap(
    blueprint,
    artifacts,
    device_ordinal=0,
    bootstrap_limits(),
  )
  let second = admit_device_worker_bootstrap(
    blueprint,
    artifacts,
    device_ordinal=0,
    bootstrap_limits(),
  )
  assert_eq(first.digest(), second.digest())
  assert_eq(first.identity(), blueprint.identity())
  assert_eq(first.device_step_limits(), blueprint.limits())
  assert_eq(blueprint.limits().max_sequence_tokens(), 8)
  assert_eq(blueprint.limits().max_token_id(), 7)
  assert_eq(first.device_ordinal(), 0)
  assert_eq(first.target(), blueprint.target())
  assert_eq(first.catalog_version(), @catalog.CatalogVersion::v3())
  assert_eq(first.module_count(), 1)
  assert_eq(first.entry_point_count(), artifacts.entry_point_count())
  assert_eq(first.step_count(), blueprint.steps().length())
  assert_eq(
    first.digest().as_hex(),
    "b2f4d230ed071ef62f0870b4e8f5edd588b38c876ce6d8b25024ef0c9ede64a2",
  )
  let startup = first.worker_startup_contract(
    @worker_protocol.ModelPlanGeneration::first(),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
    ),
    bootstrap_worker_limits(),
    bootstrap_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
  assert_eq(startup.identity(), first.identity())
  assert_eq(startup.bootstrap_digest(), first.digest())
  assert_eq(
    startup.bootstrap_source_digest().as_hex(),
    "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
  )
  assert_eq(startup.model_generation_value(), 1UL)
  assert_eq(startup.worker_limits().max_completion_slots(), 32)

  let (changed_blueprint, changed_artifacts) = bootstrap_fixture(entry_bias=100)
  let changed = admit_device_worker_bootstrap(
    changed_blueprint,
    changed_artifacts,
    device_ordinal=0,
    bootstrap_limits(),
  )
  assert_true(changed.digest() != first.digest())
  let changed_ordinal = admit_device_worker_bootstrap(
    blueprint,
    artifacts,
    device_ordinal=1,
    bootstrap_limits(),
  )
  assert_true(changed_ordinal.digest() != first.digest())
  let (changed_limits_blueprint, changed_limits_artifacts) = bootstrap_fixture(
    max_sequence_tokens=7,
  )
  let changed_limits = admit_device_worker_bootstrap(
    changed_limits_blueprint,
    changed_limits_artifacts,
    device_ordinal=0,
    bootstrap_limits(),
  )
  assert_true(changed_limits.digest() != first.digest())
}

///|
test "worker startup envelopes must fit hashed device limits" {
  let (blueprint, artifacts) = bootstrap_fixture()
  let manifest = admit_device_worker_bootstrap(
    blueprint,
    artifacts,
    device_ordinal=0,
    bootstrap_limits(),
  )
  let oversized_workers = [
    bootstrap_worker_limits(max_prefill_rows=3, max_plan_rows=3),
    bootstrap_worker_limits(max_decode_rows=3, max_plan_rows=3),
    bootstrap_worker_limits(max_plan_rows=3),
    bootstrap_worker_limits(max_plan_tokens=5),
    bootstrap_worker_limits(max_plan_pages=5),
    bootstrap_worker_limits(max_sequence_tokens=9),
    bootstrap_worker_limits(max_token_id=8),
  ]
  for oversized_worker in oversized_workers {
    try
      manifest.worker_startup_contract(
        @worker_protocol.ModelPlanGeneration::first(),
        bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
          "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
        ),
        oversized_worker,
        bootstrap_inference_limits(),
        has_predecessor=false,
        predecessor_value=0UL,
      )
    catch {
      InvalidBootstrap(RuntimeEnvelope) => ()
      _ => fail("unexpected worker-envelope error")
    } noraise {
      _ => fail("oversized worker envelope was accepted")
    }
  }
  for
    oversized_inference in [
      bootstrap_inference_limits(max_context_tokens=9),
      bootstrap_inference_limits(max_token_id=8),
    ] {
    try
      manifest.worker_startup_contract(
        @worker_protocol.ModelPlanGeneration::first(),
        bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
          "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
        ),
        bootstrap_worker_limits(),
        oversized_inference,
        has_predecessor=false,
        predecessor_value=0UL,
      )
    catch {
      InvalidBootstrap(RuntimeEnvelope) => ()
      _ => fail("unexpected inference-envelope error")
    } noraise {
      _ => fail("oversized inference envelope was accepted")
    }
  }
}

///|
test "device worker bootstrap admission is bounded before hashing" {
  let (blueprint, artifacts) = bootstrap_fixture()
  try
    admit_device_worker_bootstrap(
      blueprint,
      artifacts,
      device_ordinal=0,
      bootstrap_limits(max_steps=1),
    )
  catch {
    InvalidBootstrap(Capacity) => ()
    _ => fail("unexpected bootstrap capacity error")
  } noraise {
    _ => fail("undersized bootstrap limit was accepted")
  }
  try
    admit_device_worker_bootstrap(
      blueprint,
      artifacts,
      device_ordinal=0,
      DeviceBootstrapLimits::new(
        max_modules=4,
        max_entry_points=16,
        max_steps=32,
        max_operands_per_step=16,
        max_symbol_bytes=64,
        max_canonical_bytes=256,
      ),
    )
  catch {
    InvalidBootstrap(CanonicalBytes) => ()
    _ => fail("unexpected canonical bootstrap bound error")
  } noraise {
    _ => fail("undersized canonical-byte limit was accepted")
  }
  try
    admit_device_worker_bootstrap(
      blueprint,
      artifacts,
      device_ordinal=-1,
      bootstrap_limits(),
    )
  catch {
    InvalidBootstrap(DeviceOrdinal) => ()
    _ => fail("unexpected device-ordinal error")
  } noraise {
    _ => fail("invalid device ordinal was accepted")
  }
}
