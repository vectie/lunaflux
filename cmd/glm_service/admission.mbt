///|
fn ServicePlan::identity(self : Self) -> @identity.ModelIdentity raise {
  // Caller-supplied deployment labels. No checkpoint or executable scan.
  @identity.ModelIdentity::new(
    @identity.ContentDigest::from_sha256(self.model_label),
    @identity.PlanDigest::from_sha256(self.runtime_label),
  )
}

///|
fn ServicePlan::source(
  self : Self,
  manifest : String,
) -> @wire.EncodedBootstrapSource raise {
  @wire.admit_glm_batched_bootstrap_source(
    @wire.WorkerBootstrapRoots::new(
      model_root=self.model_root,
      kernel_root=self.kernel_root,
    ),
    @wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="model.safetensors.index.json",
      config_digest_sha256=self.model_label,
    ),
    @wire.WorkerBootstrapKernelSource::new(
      manifest_locator=manifest,
      manifest_digest_sha256=self.runtime_label,
      sampling_runtime=DeviceFullF32SamplingV1,
    ),
    @wire.WorkerBootstrapExecution::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=16,
      total_page_count=self.history().geometry().1,
    ),
    @wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1048576,
      max_weight_file_bytes=self.budget,
      max_weight_arena_bytes=self.budget - self.reserve,
      max_activation_arena_bytes=self.reserve,
      max_kv_arena_bytes=self.reserve,
      max_execution_manifest_bytes=65536L,
      max_module_bytes=self.reserve / 4L,
      max_total_module_bytes=self.reserve / 2L,
    ),
  )
}

///|
/// One admitted context bounds the useful token work of a resident request.
/// Give it that window before age-only eviction; the two-step fixture policy
/// otherwise repeatedly destroys recurrent state when one extra client waits.
/// The cap is the existing SchedulerConfig waiting-age validation ceiling,
/// not a device/kernel tuning constant. Page pressure and deadlines are still
/// handled by the scheduler's existing resource and cancellation rules.
fn service_waiting_age_steps(context : Int) -> Int {
  Int::min(context, 1_000_000)
}

///|
fn ServicePlan::blueprint(
  self : Self,
  spec : @spec.Glm53FlashSpec,
) -> @scheduler.SchedulerBlueprint raise {
  let limits = self.worker_limits(spec)
  let capacity = @resolved.resolve(
    @scheduling.new(
      step_token_budget=self.rows,
      max_active_requests=self.slots,
      max_waiting_requests=self.slots,
      prefill_chunk_tokens=self.rows,
      emergency_decode_page_reserve=self.history_capacity.decode_reserve_pages(),
      waiting_age_threshold_steps=service_waiting_age_steps(self.context),
      output_event_capacity=self.slots * 4,
    ),
    @cache.new(
      tokens_per_page=16,
      total_page_count=self.history().geometry().1,
      block_table_pages_per_request=self.context / 16,
      prefix_enabled=false,
      max_prefix_entries=0,
      max_prefix_nodes=0,
      max_prefix_tokens_per_entry=0,
      max_prefix_pages=0,
      max_active_references_per_page=1,
      max_cached_references_per_page=1,
      layout_version=1,
    ),
    @model_plan.ShapeConstraints::new(
      max_batch_rows=self.rows,
      max_sequence_tokens=self.context,
      vocabulary_size=spec.vocabulary_size(),
      hidden_size=spec.hidden_size(),
      intermediate_size=spec.dense_intermediate_size(),
      attention_head_count=spec.attention_head_count(),
      kv_head_count=spec.kv_head_count(),
      head_dimension=spec.qk_nope_head_dimension() +
        spec.qk_rope_head_dimension(),
    ),
    limits,
  )
  let generation = @protocol.ModelPlanGeneration::first()
  let recipe = @protocol.RowCapabilityRecipe::new(
    identity=self.identity(),
    model_plan_generation=generation,
    [@model_plan.KernelCapabilityId::embedding_lookup()],
    limits,
  )
  @scheduler.SchedulerBlueprint::new(
    capacity,
    self.identity(),
    generation,
    recipe,
    recipe,
    recipe,
  )
}

///|
test "service fairness derives from admitted context rather than fixture cadence" {
  assert_eq(service_waiting_age_steps(2048), 2048)
  assert_eq(service_waiting_age_steps(65536), 65536)
  assert_eq(service_waiting_age_steps(1048576), 1_000_000)
}

///|
fn prepare_service(
  plan : ServicePlan,
  manifest : String,
  root : @approved_fs.ApprovedRoot,
  kernels : @approved_fs.ApprovedRoot,
  spec : @spec.Glm53FlashSpec,
) -> @session.LunaOnlineFramedServicePreparation raise {
  let inference = plan.inference(spec)
  let source = plan.source(manifest)
  let label = @wire.WorkerBootstrapDigest::from_sha256(plan.runtime_label)
  let tokenizer = @text_input.load_checkpoint_tokenizer(
    root,
    plan.model_root,
    "tokenizer.json",
    plan.tokenizer_label,
    vocabulary_size=spec.vocabulary_size(),
    maximum_tokens=plan.context,
  )
  let startup = @wire.WorkerStartupContract::new(
    identity=plan.identity(),
    bootstrap_digest=label,
    bootstrap_source_digest=source.digest(),
    device_ordinal=0,
    model_generation_value=1UL,
    worker_limits=plan.worker_limits(spec),
    inference_limits=inference,
    has_predecessor=false,
    predecessor_value=0UL,
  )
  let framed = @framed.FramedWireLimits::new(inference, max_frame_bytes=2097152)
  let preparation_storage = plan.preparation_storage(tokenizer, inference, framed)
  @session.prepare_owned_luna_online_framed_service_approved(
    tokenizer,
    tokenizer.digest(),
    plan.identity(),
    inference,
    framed,
    plan.slots * 2,
    @admission.LunaRequestPreparationStepBudget::new(4096),
    @admission.LunaRequestPreparationWorkLimit::new(100000000UL),
    preparation_storage,
    @framed.LunaFramedEventStepBudget::new(4096),
    plan.blueprint(spec),
    @service.WorkerServiceBinding::new(
      bootstrap_digest=label,
      bootstrap_source_digest=source.digest(),
      device_ordinal=0,
      inference_limits=inference,
    ),
    @executable.verify(
      plan.executable,
      @executable.WorkerExecutableDigest::from_sha256(plan.runtime_label),
      @executable.WorkerExecutableFileLimits::new(maximum_bytes=1073741824L),
    ),
    startup,
    source,
    @worker_process.WorkerProcessLimits::new(
      max_frame_bytes=67108864,
      startup_io_timeout_millis=300000,
      io_timeout_millis=300000,
      shutdown_timeout_millis=300000,
    ),
    root,
    kernels,
    @service.WorkerRestartBackoffPolicy::new(
      initial_delay_millis=100UL,
      maximum_delay_millis=1000UL,
      stable_success_millis=10000UL,
      maximum_attempts=3,
    ),
    input_special_policy=Allow,
  )
}
