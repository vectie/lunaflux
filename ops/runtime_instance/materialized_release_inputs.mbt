///|
priv struct AdmittedMaterializedReleaseInputs {
  launch : @launch_file.LaunchFileAdmission
  runtime : AdmittedRuntimeDescriptor
  instance : AdmittedRuntimeInstance
  executable : @worker_executable_file.MaterializedWorkerExecutableEvidence
  model_root : @approved_fs.ApprovedRoot
  kernel_root : @approved_fs.ApprovedRoot
  worker_root : @approved_fs.ApprovedRoot
  bundle_sha256 : String
}

///|
fn bundle_descendant(root : StringView, suffix : StringView) -> String {
  "\{root}/\{suffix}"
}

///|
fn require_bundle_launch_binding(
  launch : @launch_file.LaunchFileAdmission,
  evidence : @deployment_bundle.DeploymentBundleEvidence,
) -> Unit raise RuntimeInstanceError {
  if launch.digest().as_hex() != evidence.launch_file_sha256() ||
    launch.descriptor_locator() != evidence.runtime_descriptor_relative() ||
    launch.descriptor_sha256() != evidence.runtime_descriptor_sha256() ||
    launch.policy_locator() != evidence.instance_policy_relative() ||
    launch.policy_sha256() != evidence.instance_policy_sha256() ||
    launch.worker_executable_sha256() != evidence.worker_executable_sha256() {
    raise fail(MaterializationBinding)
  }
}

///|
fn open_materialization_view(
  source_label : String,
  target_label : String,
  stage : RuntimeInstanceStage,
) -> (@approved_fs.ApprovedRoot, @approved_fs.ApprovedRootMaterializationView) raise RuntimeInstanceError {
  let root = @approved_fs.ApprovedRoot::open_absolute(source_label) catch {
    _ => raise fail(stage)
  }
  let view = root.materialization_view(source_label, target_label) catch {
    _ => {
      close_root(root, Cleanup)
      raise fail(MaterializationBinding)
    }
  }
  (root, view)
}

///|
fn absolute_parent(path : StringView) -> String raise RuntimeInstanceError {
  let mut slash = -1
  for index, unit in path {
    if unit == '/' {
      slash = index
    }
  }
  if slash <= 0 || slash == path.length() - 1 {
    raise fail(MaterializationBinding)
  }
  path[0:slash].to_owned()
}

///|
fn admit_materialized_release_inputs(
  bundle : MaterializedBundleAdmission,
  limits : RuntimeInstanceFileLimits,
) -> AdmittedMaterializedReleaseInputs raise RuntimeInstanceError {
  let launch_root_label = bundle_descendant(bundle.root_label, "launch-root")
  let launch = @launch_file.load_argument(
    "\{launch_root_label}#sha256=\{bundle.evidence.launch_file_sha256()}",
    limits.launch,
  ) catch {
    _ => raise fail(LaunchAdmission)
  }
  require_bundle_launch_binding(launch, bundle.evidence)
  let model_source = bundle_descendant(bundle.root_label, "model-root")
  let kernel_source = bundle_descendant(
    bundle.root_label,
    "oci-context/rootfs/opt/lunaflux/kernels",
  )
  let policy_source = bundle_descendant(bundle.root_label, "policy-root")
  let (model_root, model) = open_materialization_view(
    model_source,
    launch.model_root_label(),
    ModelRootAdmission,
  )
  let kernel_result : Result[
    (@approved_fs.ApprovedRoot, @approved_fs.ApprovedRootMaterializationView),
    RuntimeInstanceError,
  ] = Ok(
    open_materialization_view(
      kernel_source,
      launch.kernel_root_label(),
      KernelRootAdmission,
    ),
  ) catch {
    error => Err(error)
  }
  let (kernel_root, kernel) = match kernel_result {
    Ok(value) => value
    Err(error) => {
      close_root(model_root, Cleanup)
      raise error
    }
  }
  let policy_result : Result[
    (@approved_fs.ApprovedRoot, @approved_fs.ApprovedRootMaterializationView),
    RuntimeInstanceError,
  ] = Ok(
    open_materialization_view(
      policy_source,
      launch.policy_root_label(),
      PolicyRootAdmission,
    ),
  ) catch {
    error => Err(error)
  }
  let (policy_root, policy_view) = match policy_result {
    Ok(value) => value
    Err(error) => {
      close_model_kernel(model_root, kernel_root)
      raise error
    }
  }
  let approval_result : Result[
    @worker_wire.WorkerBootstrapLunaApproval?,
    RuntimeInstanceError,
  ] = Ok(approval_from_launch(launch)) catch { error => Err(error) }
  let approval = match approval_result {
    Ok(value) => value
    Err(error) => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise error
    }
  }
  let descriptor_locator = @approved_fs.ApprovedRelativeLocator::new(
    launch.descriptor_locator(),
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(DescriptorAdmission)
    }
  }
  let descriptor_digest = @descriptor_file.RuntimeDescriptorDigest::from_sha256(
    launch.descriptor_sha256(),
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, policy_root)
      raise fail(DescriptorAdmission)
    }
  }
  let runtime_result : Result[AdmittedRuntimeDescriptor, RuntimeInstanceError] = Ok(
    load_materialized_runtime_descriptor(
      launch,
      model,
      kernel,
      descriptor_locator,
      descriptor_digest,
      limits.descriptor,
      approval,
    ),
  ) catch {
    error => Err(error)
  }
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
  let policy = @instance_policy_file.load_materialized(
    policy_view,
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
  ] = Ok(load_materialized_tokenizer(model, policy)) catch {
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
  let worker_source = bundle_descendant(
    bundle.root_label,
    "oci-context/rootfs/opt/lunaflux/bin",
  )
  let worker_target_parent = absolute_parent(launch.worker_executable_path())
  let worker_result : Result[
    (@approved_fs.ApprovedRoot, @approved_fs.ApprovedRootMaterializationView),
    RuntimeInstanceError,
  ] = Ok(
    open_materialization_view(
      worker_source,
      worker_target_parent,
      WorkerExecutableAdmission,
    ),
  ) catch {
    error => Err(error)
  }
  let (worker_root, worker) = match worker_result {
    Ok(value) => value
    Err(error) => {
      close_model_kernel(model_root, kernel_root)
      raise error
    }
  }
  let executable_digest = @worker_executable_file.WorkerExecutableDigest::from_sha256(
    launch.worker_executable_sha256(),
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, worker_root)
      raise fail(WorkerExecutableAdmission)
    }
  }
  let executable = @worker_executable_file.verify_materialized(
    worker,
    "lunaflux-device-worker",
    launch.worker_executable_path(),
    executable_digest,
    limits.executable,
  ) catch {
    _ => {
      close_three_roots(model_root, kernel_root, worker_root)
      raise fail(WorkerExecutableAdmission)
    }
  }
  {
    launch,
    runtime,
    instance,
    executable,
    model_root,
    kernel_root,
    worker_root,
    bundle_sha256: bundle.evidence.digest().as_hex(),
  }
}
