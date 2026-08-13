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
fn bootstrap_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens=8,
    max_token_id=7,
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
fn bootstrap_fixture(
  digest_head? : Char = '0',
  entry_bias? : Int = 0,
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
    blueprint_limits(),
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
    bootstrap_limits(),
  )
  let second = admit_device_worker_bootstrap(
    blueprint,
    artifacts,
    bootstrap_limits(),
  )
  assert_eq(first.digest(), second.digest())
  assert_eq(first.identity(), blueprint.identity())
  assert_eq(first.target(), blueprint.target())
  assert_eq(first.catalog_version(), @catalog.CatalogVersion::v3())
  assert_eq(first.module_count(), 1)
  assert_eq(first.entry_point_count(), artifacts.entry_point_count())
  assert_eq(first.step_count(), blueprint.steps().length())
  assert_eq(
    first.digest().as_hex(),
    "d404e6ee3e3cc857240f785399d581de04c47433ec5dbc4552a2885b8d212f67",
  )
  let startup = first.worker_startup_contract(
    @worker_protocol.ModelPlanGeneration::first(),
    step_protocol_limits(),
    bootstrap_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
  assert_eq(startup.identity(), first.identity())
  assert_eq(startup.bootstrap_digest(), first.digest())
  assert_eq(startup.model_generation_value(), 1UL)

  let (changed_blueprint, changed_artifacts) = bootstrap_fixture(entry_bias=100)
  let changed = admit_device_worker_bootstrap(
    changed_blueprint,
    changed_artifacts,
    bootstrap_limits(),
  )
  assert_true(changed.digest() != first.digest())
}

///|
test "device worker bootstrap admission is bounded before hashing" {
  let (blueprint, artifacts) = bootstrap_fixture()
  try
    admit_device_worker_bootstrap(
      blueprint,
      artifacts,
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
}
