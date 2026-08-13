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
    "a55eaf90b9e52e5de7b21d72fd48b1740400d51c7c9575daad5d923ba6aa2a2c",
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
