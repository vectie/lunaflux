///|
/// Qualification is deliberately narrower than general listener admission.
/// Deployment exposure remains governed by the authenticated production
/// protocol policy and is not inferred here.
fn loopback_openai_host(host : String) -> Bool {
  host == "127.0.0.1" || host == "localhost" || host == "::1"
}

///|
fn private_network_openai_host(host : String) -> Bool {
  host == "0.0.0.0" || host == "::"
}

///|
/// Executes exactly one loader selected by the closed authenticated recipe.
/// An error from that loader is propagated directly; the other loader is
/// never probed as fallback.
fn route_runtime_recipe(
  recipe : @launch_file.LaunchRuntimeRecipe,
  legacy_loader : () -> AdmittedRuntimeDescriptor raise RuntimeInstanceError,
  i8_loader : () -> AdmittedRuntimeDescriptor raise RuntimeInstanceError,
  mistral_loader : () -> AdmittedRuntimeDescriptor raise RuntimeInstanceError,
  tensor_parallel_loader : () -> AdmittedRuntimeDescriptor raise RuntimeInstanceError,
  fp8_loader : () -> AdmittedRuntimeDescriptor raise RuntimeInstanceError,
) -> AdmittedRuntimeDescriptor raise RuntimeInstanceError {
  match recipe {
    DenseLlamaPagedAotV5 => legacy_loader()
    DenseLlamaI8PagedAotV6 => i8_loader()
    DenseMistralBf16PagedAotV7 => mistral_loader()
    DenseLlamaTensorParallelPagedAotApprovedV8 => tensor_parallel_loader()
    DenseLlamaFp8ReusablePagedAotApprovedV9 => fp8_loader()
  }
}

///|
fn load_runtime_descriptor(
  launch : @launch_file.LaunchFileAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  descriptor_digest : @descriptor_file.RuntimeDescriptorDigest,
  limits : @descriptor_file.RuntimeDescriptorFileLimits,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval?,
  luna_verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> AdmittedRuntimeDescriptor raise RuntimeInstanceError {
  route_runtime_recipe(
    launch.runtime_recipe(),
    () => {
      LegacyRuntimeDescriptor(
        @descriptor_file.load(
          model_root,
          kernel_root,
          launch.model_root_label(),
          launch.kernel_root_label(),
          descriptor_locator,
          descriptor_digest,
          limits,
          luna_approval~,
          luna_verifier~,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      // Launch admission has already rejected an approval that recipe v6
      // cannot encode. No fallback to the legacy loader is permitted.
      if luna_approval is Some(_) {
        raise fail(LaunchAdmission)
      }
      I8RuntimeDescriptor(
        @descriptor_file.load_i8_v2(
          model_root,
          kernel_root,
          launch.model_root_label(),
          launch.kernel_root_label(),
          descriptor_locator,
          descriptor_digest,
          limits,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      if luna_approval is Some(_) {
        raise fail(LaunchAdmission)
      }
      LegacyRuntimeDescriptor(
        @descriptor_file.load_mistral_v1(
          model_root,
          kernel_root,
          launch.model_root_label(),
          launch.kernel_root_label(),
          descriptor_locator,
          descriptor_digest,
          limits,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      guard luna_approval is Some(approval) else { raise fail(LaunchAdmission) }
      guard luna_verifier is Some(verifier) else { raise fail(LaunchAdmission) }
      TensorParallelRuntimeDescriptor(
        @descriptor_file.load_tensor_parallel(
          model_root,
          kernel_root,
          launch.model_root_label(),
          launch.kernel_root_label(),
          descriptor_locator,
          descriptor_digest,
          limits,
          approval,
          verifier,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      guard luna_approval is Some(approval) else { raise fail(LaunchAdmission) }
      guard luna_verifier is Some(verifier) else { raise fail(LaunchAdmission) }
      Fp8RuntimeDescriptor(
        @descriptor_file.load_fp8_reusable_v3(
          model_root,
          kernel_root,
          launch.model_root_label(),
          launch.kernel_root_label(),
          descriptor_locator,
          descriptor_digest,
          limits,
          approval,
          verifier,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
  )
}

///|
/// Executes the same closed recipe route against typed offline
/// source-capability/target-label views. The source roots are pinned and
/// no-follow; the reconstructed bootstrap retains only the final target labels.
fn load_materialized_runtime_descriptor(
  launch : @launch_file.LaunchFileAdmission,
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  descriptor_digest : @descriptor_file.RuntimeDescriptorDigest,
  limits : @descriptor_file.RuntimeDescriptorFileLimits,
  luna_approval : @worker_wire.WorkerBootstrapLunaApproval?,
) -> AdmittedRuntimeDescriptor raise RuntimeInstanceError {
  route_runtime_recipe(
    launch.runtime_recipe(),
    () => {
      LegacyRuntimeDescriptor(
        @descriptor_file.load_materialized(
          model,
          kernel,
          descriptor_locator,
          descriptor_digest,
          limits,
          luna_approval~,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      if luna_approval is Some(_) {
        raise fail(LaunchAdmission)
      }
      I8RuntimeDescriptor(
        @descriptor_file.load_i8_v2_materialized(
          model, kernel, descriptor_locator, descriptor_digest, limits,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      if luna_approval is Some(_) {
        raise fail(LaunchAdmission)
      }
      LegacyRuntimeDescriptor(
        @descriptor_file.load_mistral_v1_materialized(
          model, kernel, descriptor_locator, descriptor_digest, limits,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      guard luna_approval is Some(approval) else { raise fail(LaunchAdmission) }
      TensorParallelRuntimeDescriptor(
        @descriptor_file.load_tensor_parallel_materialized(
          model, kernel, descriptor_locator, descriptor_digest, limits, approval,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
    () => {
      guard luna_approval is Some(approval) else { raise fail(LaunchAdmission) }
      Fp8RuntimeDescriptor(
        @descriptor_file.load_fp8_reusable_v3_materialized(
          model, kernel, descriptor_locator, descriptor_digest, limits, approval,
        ) catch {
          _ => raise fail(DescriptorAdmission)
        },
      )
    },
  )
}

///|
fn admit_runtime_instance(
  runtime : AdmittedRuntimeDescriptor,
  policy : @instance_policy_file.InstancePolicyAdmission,
  tokenizer : @json_file.TokenizerFileAdmission,
) -> AdmittedRuntimeInstance raise RuntimeInstanceError {
  match runtime {
    LegacyRuntimeDescriptor(descriptor) =>
      LegacyRuntimeInstance(
        @instance_admission.admit(descriptor, policy, tokenizer) catch {
          _ => raise fail(InstanceAdmission)
        },
      )
    I8RuntimeDescriptor(descriptor) =>
      I8RuntimeInstance(
        @instance_admission.admit_i8(descriptor, policy, tokenizer) catch {
          _ => raise fail(InstanceAdmission)
        },
      )
    Fp8RuntimeDescriptor(descriptor) =>
      Fp8RuntimeInstance(
        @instance_admission.admit_fp8_reusable(descriptor, policy, tokenizer) catch {
          _ => raise fail(InstanceAdmission)
        },
      )
    TensorParallelRuntimeDescriptor(descriptor) =>
      TensorParallelRuntimeInstance(
        @instance_admission.admit_tensor_parallel(descriptor, policy, tokenizer) catch {
          _ => raise fail(InstanceAdmission)
        },
      )
  }
}

///|
fn AdmittedRuntimeDescriptor::admit_physical(
  self : AdmittedRuntimeDescriptor,
) -> Unit raise RuntimeInstanceError {
  match self {
    LegacyRuntimeDescriptor(runtime) => admit_physical_device(runtime)
    I8RuntimeDescriptor(runtime) => admit_i8_physical_device(runtime)
    Fp8RuntimeDescriptor(_) => ()
    TensorParallelRuntimeDescriptor(_) => ()
  }
}

///|
fn AdmittedRuntimeInstance::service_envelope(
  self : AdmittedRuntimeInstance,
) -> RuntimeServiceEnvelope raise RuntimeInstanceError {
  let (
    config,
    transport,
    telemetry_capacity,
    connection_capacity,
    request_work_units,
    event_work_units,
    inference,
  ) = match self {
    LegacyRuntimeInstance(instance) =>
      (
        instance.service_config(),
        instance.transport_limits(),
        instance.telemetry_capacity(),
        instance.preparation_lane_count(),
        instance.preparation_step_budget().work_units(),
        instance.event_step_budget().work_units(),
        instance.inference_limits(),
      )
    I8RuntimeInstance(instance) =>
      (
        instance.service_config(),
        instance.transport_limits(),
        instance.telemetry_capacity(),
        instance.preparation_lane_count(),
        instance.preparation_step_budget().work_units(),
        instance.event_step_budget().work_units(),
        instance.inference_limits(),
      )
    Fp8RuntimeInstance(instance) =>
      (
        instance.service_config(),
        instance.transport_limits(),
        instance.telemetry_capacity(),
        instance.preparation_lane_count(),
        instance.preparation_step_budget().work_units(),
        instance.event_step_budget().work_units(),
        instance.inference_limits(),
      )
    TensorParallelRuntimeInstance(instance) =>
      (
        instance.service_config(),
        instance.transport_limits(),
        instance.telemetry_capacity(),
        instance.preparation_lane_count(),
        instance.preparation_step_budget().work_units(),
        instance.event_step_budget().work_units(),
        instance.inference_limits(),
      )
  }
  let addr = @socket.Addr::parse(
    "\{config.listen_host()}:\{config.listen_port()}",
  ) catch {
    _ => raise fail(ServicePreparation)
  }
  let protocol = self.external_protocol()
  let openai_listener_allowed = match protocol.transport_security() {
    LoopbackPlaintext => loopback_openai_host(config.listen_host())
    PrivateNetworkPlaintext =>
      protocol.mode() == OpenAiResponsesV1 &&
      private_network_openai_host(config.listen_host())
    DeploymentTls => false
  }
  {
    addr,
    transport,
    telemetry_capacity,
    connection_capacity,
    request_work_units,
    event_work_units,
    receipt_deadline: @inference.DeadlineBudget::from_milliseconds(
      inference.max_deadline_millis(),
      inference,
    ) catch {
      _ => raise fail(ServicePreparation)
    },
    loopback_openai_allowed: loopback_openai_host(config.listen_host()),
    openai_listener_allowed,
  }
}

///|
fn AdmittedRuntimeInstance::external_protocol(
  self : AdmittedRuntimeInstance,
) -> @instance_policy_file.ExternalProtocolPolicyClaims {
  match self {
    LegacyRuntimeInstance(instance) => instance.external_protocol()
    I8RuntimeInstance(instance) => instance.external_protocol()
    Fp8RuntimeInstance(instance) => instance.external_protocol()
    TensorParallelRuntimeInstance(instance) => instance.external_protocol()
  }
}

///|
fn AdmittedRuntimeInstance::openai_service_policy(
  self : AdmittedRuntimeInstance,
) -> @instance_policy_file.OpenAiServicePolicyClaims? {
  match self {
    LegacyRuntimeInstance(instance) => instance.openai_service_policy()
    I8RuntimeInstance(instance) => instance.openai_service_policy()
    Fp8RuntimeInstance(instance) => instance.openai_service_policy()
    TensorParallelRuntimeInstance(instance) => instance.openai_service_policy()
  }
}

///|
fn AdmittedRuntimeInstance::inference_limits(
  self : AdmittedRuntimeInstance,
) -> @inference.InferenceLimits {
  match self {
    LegacyRuntimeInstance(instance) => instance.inference_limits()
    I8RuntimeInstance(instance) => instance.inference_limits()
    Fp8RuntimeInstance(instance) => instance.inference_limits()
    TensorParallelRuntimeInstance(instance) => instance.inference_limits()
  }
}

///|
fn AdmittedRuntimeInstance::prepare_owned_service(
  self : AdmittedRuntimeInstance,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> @online_session.LunaOnlineFramedServicePreparation raise RuntimeInstanceError {
  match self {
    LegacyRuntimeInstance(instance) => {
      let runtime = instance.runtime_descriptor()
      @online_session.prepare_owned_luna_online_framed_service_approved(
        instance.tokenizer(),
        instance.tokenizer_digest(),
        runtime.model_identity(),
        instance.worker_binding().inference_limits(),
        instance.framed_limits(),
        instance.preparation_lane_count(),
        instance.preparation_step_budget(),
        instance.preparation_work_limit(),
        instance.preparation_storage_budget(),
        instance.event_step_budget(),
        instance.scheduler_blueprint(),
        instance.worker_binding(),
        executable,
        runtime.startup_contract(),
        runtime.bootstrap_source(),
        instance.worker_process_limits(),
        model_root,
        kernel_root,
        instance.restart_policy(),
      ) catch {
        _ => raise fail(ServicePreparation)
      }
    }
    I8RuntimeInstance(instance) =>
      @i8_release_worker.prepare_owned_online_framed(
        instance, executable, model_root, kernel_root,
      ) catch {
        _ => raise fail(ServicePreparation)
      }
    Fp8RuntimeInstance(instance) =>
      @fp8_release_worker.prepare_owned_online_framed_v3(
        instance, executable, model_root, kernel_root,
      ) catch {
        _ => raise fail(ServicePreparation)
      }
    TensorParallelRuntimeInstance(instance) => {
      let runtime = instance.runtime_descriptor()
      let inputs = runtime.planning_inputs(
        instance.rank_group_process_limits() catch {
          _ => raise fail(ServicePreparation)
        },
        executable.activation_path(),
        executable.digest().as_hex(),
      ) catch {
        _ => raise fail(ServicePreparation)
      }
      @online_session.prepare_owned_tensor_parallel_luna_online_framed_service(
        instance.tokenizer(),
        instance.tokenizer_digest(),
        runtime.model_plan().identity(),
        instance.inference_limits(),
        instance.framed_limits(),
        instance.preparation_lane_count(),
        instance.preparation_step_budget(),
        instance.preparation_work_limit(),
        instance.preparation_storage_budget(),
        instance.event_step_budget(),
        instance.scheduler_blueprint(),
        inputs,
        executable,
        model_root,
        kernel_root,
        instance.restart_policy(),
      ) catch {
        _ => raise fail(ServicePreparation)
      }
    }
  }
}
