///|
fn close_promotion_verifier(
  verifier : @luna_capability_manifest.LunaStartupExternalApprovalHandoff?,
) -> Unit {
  if verifier is Some(owner) {
    owner.close()
  }
}

///|
/// Performs the authority-bearing but device-free release join shared by the
/// offline preflight and live preparation. Policy authority is consumed here;
/// the returned pair remains caller-owned until either path closes or transfers
/// it.
fn admit_release_inputs(
  model_argument : StringView,
  limits : RuntimeInstanceFileLimits,
  luna_verifier? : @luna_capability_manifest.LunaStartupExternalApprovalHandoff? = None,
) -> AdmittedReleaseInputs raise RuntimeInstanceError {
  let launch = @launch_file.load_argument(model_argument, limits.launch) catch {
    _ => {
      close_promotion_verifier(luna_verifier)
      raise fail(LaunchAdmission)
    }
  }
  let model_label = launch.model_root_label()
  let kernel_label = launch.kernel_root_label()
  let policy_label = launch.policy_root_label()
  let model_root = @approved_fs.ApprovedRoot::open_absolute(model_label) catch {
    _ => {
      close_promotion_verifier(luna_verifier)
      raise fail(ModelRootAdmission)
    }
  }
  let kernel_root = @approved_fs.ApprovedRoot::open_absolute(kernel_label) catch {
    _ => {
      close_promotion_verifier(luna_verifier)
      close_root(model_root, Cleanup)
      raise fail(KernelRootAdmission)
    }
  }
  let policy_root = @approved_fs.ApprovedRoot::open_absolute(policy_label) catch {
    _ => {
      close_promotion_verifier(luna_verifier)
      close_model_kernel(model_root, kernel_root)
      raise fail(PolicyRootAdmission)
    }
  }
  let approval_result : Result[
    @worker_wire.WorkerBootstrapLunaApproval?,
    RuntimeInstanceError,
  ] = Ok(approval_from_launch(launch)) catch { error => Err(error) }
  let approval = match approval_result {
    Ok(value) => value
    Err(error) => {
      close_promotion_verifier(luna_verifier)
      close_three_roots(model_root, kernel_root, policy_root)
      raise error
    }
  }
  match (approval, luna_verifier) {
    (None, None) | (Some(_), Some(_)) => ()
    _ => {
      close_promotion_verifier(luna_verifier)
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(LaunchAdmission)
    }
  }
  let descriptor_locator = @approved_fs.ApprovedRelativeLocator::new(
    launch.descriptor_locator(),
  ) catch {
    _ => {
      close_promotion_verifier(luna_verifier)
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(DescriptorAdmission)
    }
  }
  let descriptor_digest = @descriptor_file.RuntimeDescriptorDigest::from_sha256(
    launch.descriptor_sha256(),
  ) catch {
    _ => {
      close_promotion_verifier(luna_verifier)
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(DescriptorAdmission)
    }
  }
  let runtime_result : Result[AdmittedRuntimeDescriptor, RuntimeInstanceError] = Ok(
    load_runtime_descriptor(
      launch,
      model_root,
      kernel_root,
      descriptor_locator,
      descriptor_digest,
      limits.descriptor,
      approval,
      luna_verifier~,
    ),
  ) catch {
    error => Err(error)
  }
  close_promotion_verifier(luna_verifier)
  let runtime = match runtime_result {
    Ok(value) => value
    Err(error) => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise error
    }
  }
  let policy_locator = @approved_fs.ApprovedRelativeLocator::new(
    launch.policy_locator(),
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(PolicyAdmission)
    }
  }
  let policy_digest = @instance_policy_file.InstancePolicyDigest::from_sha256(
    launch.policy_sha256(),
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(PolicyAdmission)
    }
  }
  let policy = @instance_policy_file.load(
    policy_root,
    policy_label,
    policy_locator,
    policy_digest,
    limits.policy,
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(PolicyAdmission)
    }
  }
  let tokenizer_result : Result[
    @json_file.TokenizerFileAdmission,
    RuntimeInstanceError,
  ] = Ok(load_tokenizer(model_root, model_label, policy)) catch {
    error => Err(error)
  }
  let tokenizer = match tokenizer_result {
    Ok(value) => value
    Err(error) => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise error
    }
  }
  let instance_result : Result[AdmittedRuntimeInstance, RuntimeInstanceError] = Ok(
    admit_runtime_instance(runtime, policy, tokenizer),
  ) catch {
    error => Err(error)
  }
  let instance = match instance_result {
    Ok(value) => value
    Err(error) => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise error
    }
  }
  policy_root.close() catch {
    _ => {
      close_model_kernel(model_root, kernel_root)
      raise fail(PolicyClose)
    }
  }
  { launch, runtime, instance, model_root, kernel_root }
}
