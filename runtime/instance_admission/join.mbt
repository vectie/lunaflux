///| Purely join independently admitted runtime, policy, and tokenizer evidence.

///|
/// No filesystem access, device probe, process creation, socket activation, or
/// mutable owner construction occurs here.
fn derive_instance(
  descriptor_digest : @descriptor_file.RuntimeDescriptorDigest,
  model_root : StringView,
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  inference : @inference.InferenceLimits,
  worker : @worker_protocol.WorkerProtocolLimits,
  model_generation_value : UInt64,
  worker_binding : @worker_service.WorkerServiceBinding?,
  policy : @instance_policy_file.InstancePolicyAdmission,
  tokenizer_file : @json_file.TokenizerFileAdmission,
) -> RuntimeInstanceDerived raise RuntimeInstanceAdmissionError {
  require_runtime_digest(policy.runtime_descriptor_sha256(), descriptor_digest)
  let (tokenizer, tokenizer_digest) = admit_tokenizer_binding(
    model_root,
    policy.tokenizer(),
    tokenizer_file,
  )
  require_tokenizer_identity(policy.tokenizer().sha256(), tokenizer_digest)
  let shapes = plan.shape_constraints()
  require_tokenizer_envelope(
    tokenizer.vocab_size(),
    shapes.vocabulary_size(),
    tokenizer.limits().max_input_bytes,
    inference.max_text_bytes(),
    tokenizer.limits().max_output_tokens,
    inference.max_input_tokens(),
    tokenizer.limits().max_decoded_bytes,
    inference.max_decoded_delta_bytes(),
  )
  let scheduler = admit_scheduler(policy.scheduler())
  let cache = admit_cache(policy.cache())
  require_layout_geometry(
    cache.tokens_per_page(),
    cache.total_page_count(),
    cache.layout_version(),
    layout.tokens_per_page(),
    layout.total_page_count(),
    layout.stable_layout_version(),
  )
  if cache.prefix_enabled() &&
    cache.max_prefix_scope_bytes() < inference.max_cache_scope_bytes() {
    raise reject(Layout)
  }
  let service = admit_service(policy.service())
  let protocol_admitted = external_protocol_admitted(
    policy.external_protocol(),
    service,
    openai_service_available=policy.openai_service() is Some(_),
  )
  if !protocol_admitted {
    raise reject(ExternalProtocol)
  }
  let worker_process = admit_worker_process(policy.worker_process(), worker)
  let restart = admit_restart(policy.restart())
  let transport = admit_transport(policy.transport())
  let maximum_wait = @online_session.admitted_maximum_transport_wait_millis(
    inference,
  ) catch {
    _ => raise reject(Transport)
  }
  transport.require_fits_transport_wait(maximum_wait) catch {
    _ => raise reject(Transport)
  }
  let preparation = policy.preparation()
  let framed = @framed_wire.FramedWireLimits::new(
    inference,
    max_frame_bytes=preparation.framed_max_frame_bytes(),
  ) catch {
    _ => raise reject(FramedWire)
  }
  require_service_frame_bytes(
    service.max_request_bytes(),
    framed.max_frame_bytes(),
  )
  let capacity = @runtime_resolved.resolve(scheduler, cache, shapes, worker) catch {
    _ => raise reject(Capacity)
  }
  require_preparation_lanes(
    preparation.lane_count(),
    capacity.total_request_slots(),
  )
  let preparation_step = @request_admission.LunaRequestPreparationStepBudget::new(
    preparation.step_work_units(),
  ) catch {
    _ => raise reject(Preparation)
  }
  let preparation_work_limit = @request_admission.LunaRequestPreparationWorkLimit::new(
    preparation.total_work_units(),
  ) catch {
    _ => raise reject(Preparation)
  }
  let preparation_storage = @request_admission.LunaRequestPreparationStorageBudget::new(
    int_cells=preparation.storage_int_cells(),
    byte_cells=preparation.storage_byte_cells(),
    reference_cells=preparation.storage_reference_cells(),
  ) catch {
    _ => raise reject(Preparation)
  }
  preparation_storage.require_fits(
    tokenizer,
    inference,
    framed,
    preparation.lane_count(),
  ) catch {
    _ => raise reject(Preparation)
  }
  let event_step = @framed_wire.LunaFramedEventStepBudget::new(
    preparation.event_work_units(),
  ) catch {
    _ => raise reject(Preparation)
  }
  let telemetry_capacity = policy.telemetry().instance_log_capacity()
  ignore(
    @instance.LunaInstanceLog::required_int_cells(telemetry_capacity) catch {
      _ => raise reject(Telemetry)
    },
  )
  ignore(
    @instance.LunaInstanceLog::required_uint64_cells(telemetry_capacity) catch {
      _ => raise reject(Telemetry)
    },
  )
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    model_generation_value,
  ) catch {
    _ => raise reject(CapabilityRecipe)
  }
  let recipe = @worker_protocol.RowCapabilityRecipe::new(
    identity=plan.identity(),
    model_plan_generation=generation,
    plan.required_capabilities(),
    worker,
  ) catch {
    _ => raise reject(CapabilityRecipe)
  }
  let scheduler_blueprint = @core.SchedulerBlueprint::new(
    capacity,
    plan.identity(),
    generation,
    recipe,
    recipe,
    recipe,
  ) catch {
    _ => raise reject(SchedulerBlueprint)
  }
  require_blueprint_capacity(
    capacity.total_request_slots(),
    scheduler_blueprint.request_slot_capacity(),
  )
  {
    policy_digest: policy.digest(),
    tokenizer,
    tokenizer_digest,
    capacity,
    scheduler_blueprint,
    worker_binding,
    service,
    external_protocol: policy.external_protocol(),
    openai_service: policy.openai_service(),
    inference,
    worker_process,
    restart,
    transport,
    framed,
    preparation_lane_count: preparation.lane_count(),
    preparation_step,
    preparation_work_limit,
    preparation_storage,
    event_step,
    telemetry_capacity,
  }
}

///|
/// Purely joins independently admitted legacy runtime, policy, and tokenizer
/// evidence. Legacy API and ownership remain unchanged.
pub fn admit(
  runtime : @descriptor_file.RuntimeDescriptorAdmission,
  policy : @instance_policy_file.InstancePolicyAdmission,
  tokenizer_file : @json_file.TokenizerFileAdmission,
) -> RuntimeInstanceAdmission raise RuntimeInstanceAdmissionError {
  let execution = runtime.execution()
  let derived = derive_instance(
    runtime.descriptor_digest(),
    runtime.bootstrap_source().roots().model_root(),
    execution.model_plan(),
    execution.layout(),
    runtime.startup_contract().inference_limits(),
    runtime.startup_contract().worker_limits(),
    runtime.startup_contract().model_generation_value(),
    Some(
      @worker_service.WorkerServiceBinding::new(
        bootstrap_digest=runtime.startup_contract().bootstrap_digest(),
        bootstrap_source_digest=runtime
          .startup_contract()
          .bootstrap_source_digest(),
        device_ordinal=runtime.startup_contract().device_ordinal(),
        inference_limits=runtime.startup_contract().inference_limits(),
        luna_authorization=execution.luna_authorization(),
      ) catch {
        _ => raise reject(WorkerBinding)
      },
    ),
    policy,
    tokenizer_file,
  )
  { runtime, core: derived }
}

///|
/// Purely joins the strict runtime-v2 I8 descriptor with independently
/// digest-pinned policy and tokenizer evidence. No filesystem, process,
/// device, socket, mutable scheduler, or readiness authority is created.
pub fn admit_i8(
  runtime : @descriptor_file.I8RuntimeDescriptorAdmission,
  policy : @instance_policy_file.InstancePolicyAdmission,
  tokenizer_file : @json_file.TokenizerFileAdmission,
) -> I8RuntimeInstanceAdmission raise RuntimeInstanceAdmissionError {
  let execution = runtime.execution()
  let derived = derive_instance(
    runtime.descriptor_digest(),
    runtime.bootstrap_source().roots().model_root(),
    execution.model_plan(),
    execution.layout(),
    runtime.startup_contract().inference_limits(),
    runtime.startup_contract().worker_limits(),
    runtime.startup_contract().model_generation_value(),
    Some(
      @worker_service.WorkerServiceBinding::new(
        bootstrap_digest=runtime.startup_contract().bootstrap_digest(),
        bootstrap_source_digest=runtime
          .startup_contract()
          .bootstrap_source_digest(),
        device_ordinal=runtime.startup_contract().device_ordinal(),
        inference_limits=runtime.startup_contract().inference_limits(),
      ) catch {
        _ => raise reject(WorkerBinding)
      },
    ),
    policy,
    tokenizer_file,
  )
  { runtime, core: derived }
}

///|
pub fn admit_fp8_reusable(
  runtime : @descriptor_file.Fp8RuntimeDescriptorAdmission,
  policy : @instance_policy_file.InstancePolicyAdmission,
  tokenizer_file : @json_file.TokenizerFileAdmission,
) -> Fp8RuntimeInstanceAdmission raise RuntimeInstanceAdmissionError {
  let execution = runtime.execution()
  let derived = derive_instance(
    runtime.descriptor_digest(),
    runtime.bootstrap_source().roots().model_root(),
    execution.model_plan(),
    execution.kv_layout(),
    runtime.startup_contract().inference_limits(),
    runtime.startup_contract().worker_limits(),
    runtime.startup_contract().model_generation_value(),
    Some(
      @worker_service.WorkerServiceBinding::new(
        bootstrap_digest=runtime.startup_contract().bootstrap_digest(),
        bootstrap_source_digest=runtime
          .startup_contract()
          .bootstrap_source_digest(),
        device_ordinal=runtime.startup_contract().device_ordinal(),
        inference_limits=runtime.startup_contract().inference_limits(),
        fp8_authorization=Some(execution.release_authority()),
      ) catch {
        _ => raise reject(WorkerBinding)
      },
    ),
    policy,
    tokenizer_file,
  )
  { runtime, core: derived }
}

///|
/// Purely joins the exact tensor-parallel runtime with independently pinned
/// policy and tokenizer evidence. Rank processes and device resources remain
/// unopened until the group-owner preparation boundary.
pub fn admit_tensor_parallel(
  runtime : @descriptor_file.TensorParallelRuntimeDescriptorAdmission,
  policy : @instance_policy_file.InstancePolicyAdmission,
  tokenizer_file : @json_file.TokenizerFileAdmission,
) -> TensorParallelRuntimeInstanceAdmission raise RuntimeInstanceAdmissionError {
  let source = runtime.bootstrap_source()
  let core = derive_instance(
    runtime.descriptor_digest(),
    source.roots().model_root(),
    runtime.model_plan(),
    runtime.kv_layout(),
    runtime.inference_limits(),
    runtime.worker_limits(),
    runtime.model_generation_value(),
    None,
    policy,
    tokenizer_file,
  )
  { runtime, core }
}
