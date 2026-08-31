///|
/// Authority retained between the common semantic join and either offline
/// evidence publication or live device preparation.
priv struct AdmittedReleaseInputs {
  launch : @launch_file.LaunchFileAdmission
  runtime : AdmittedRuntimeDescriptor
  instance : AdmittedRuntimeInstance
  model_root : @approved_fs.ApprovedRoot
  kernel_root : @approved_fs.ApprovedRoot
}

///|
/// Root-free proof that launch, descriptor, model, kernels, policy, tokenizer,
/// and worker executable were joined by the production admission path. This
/// proof grants no filesystem, process, device, compiler, or listener authority.
pub struct RuntimeReleaseAdmission {
  priv runtime_recipe : String
  priv model_content_sha256 : String
  priv model_plan_sha256 : String
  priv tokenizer_sha256 : String
  priv worker_executable_sha256 : String
  priv sampling_runtime : @worker_wire.WorkerBootstrapSamplingRuntime
  priv has_fused_runtime : Bool
  priv evidence : String
  priv evidence_sha256 : String
}

///|
/// Preserves the typed release-admission failure for CLI and deployment-tool
/// diagnostics. The raising API remains the production control path.
pub fn preflight_release_result(
  deployment : StringView,
  limits : RuntimeInstanceFileLimits,
) -> Result[RuntimeReleaseAdmission, RuntimeInstanceError] {
  Ok(preflight_release(deployment, limits)) catch {
    error => Err(error)
  }
}

///|
fn runtime_recipe_name(recipe : @launch_file.LaunchRuntimeRecipe) -> String {
  match recipe {
    DenseLlamaPagedAotV5 => "dense_llama_paged_aot_v5"
    DenseLlamaI8PagedAotV6 => "dense_llama_i8_paged_aot_v6"
    DenseMistralBf16PagedAotV7 => "dense_mistral_bf16_paged_aot_v7"
    DenseQwen3Bf16PagedAotV12 => "dense_qwen3_bf16_paged_aot_v12"
    DenseLlamaTensorParallelPagedAotApprovedV8 =>
      "dense_llama_tensor_parallel_paged_aot_approved_v8"
    DenseLlamaFp8ReusablePagedAotApprovedV9 =>
      "dense_llama_fp8_reusable_paged_aot_approved_v9"
  }
}

///|
fn AdmittedRuntimeDescriptor::descriptor_sha256(
  self : AdmittedRuntimeDescriptor,
) -> String {
  match self {
    LegacyRuntimeDescriptor(value) => value.descriptor_digest().as_hex()
    I8RuntimeDescriptor(value) => value.descriptor_digest().as_hex()
    Fp8RuntimeDescriptor(value) => value.descriptor_digest().as_hex()
    TensorParallelRuntimeDescriptor(value) => value.descriptor_digest().as_hex()
  }
}

///|
fn AdmittedRuntimeDescriptor::release_model_identity(
  self : AdmittedRuntimeDescriptor,
) -> (String, String) {
  let identity = match self {
    LegacyRuntimeDescriptor(value) => value.model_identity()
    I8RuntimeDescriptor(value) => value.model_identity()
    Fp8RuntimeDescriptor(value) => value.model_identity()
    TensorParallelRuntimeDescriptor(value) => value.model_plan().identity()
  }
  (identity.content().as_hex(), identity.plan().as_hex())
}

///|
fn AdmittedRuntimeDescriptor::release_bootstrap(
  self : AdmittedRuntimeDescriptor,
) -> (String, String) {
  match self {
    LegacyRuntimeDescriptor(value) =>
      (
        value.bootstrap_digest().as_hex(),
        value.bootstrap_source().digest().as_hex(),
      )
    I8RuntimeDescriptor(value) =>
      (
        value.bootstrap_digest().as_hex(),
        value.bootstrap_source().digest().as_hex(),
      )
    Fp8RuntimeDescriptor(value) =>
      (
        value.bootstrap_digest().as_hex(),
        value.bootstrap_source().digest().as_hex(),
      )
    TensorParallelRuntimeDescriptor(value) =>
      // A final group-contract digest does not exist until physical topology,
      // rank-local execution plans, rendezvous, and generation are admitted by
      // tensor_parallel_group_transport. Preflight therefore binds the real
      // canonical source digest and never fabricates a zero/final group claim.
      (
        value.bootstrap_source().digest().as_hex(),
        value.bootstrap_source().digest().as_hex(),
      )
  }
}

///|
fn AdmittedRuntimeDescriptor::release_execution_extensions(
  self : AdmittedRuntimeDescriptor,
) -> (@worker_wire.WorkerBootstrapSamplingRuntime, Bool) {
  let source = match self {
    LegacyRuntimeDescriptor(value) => value.bootstrap_source()
    I8RuntimeDescriptor(value) => value.bootstrap_source()
    Fp8RuntimeDescriptor(value) => value.bootstrap_source()
    TensorParallelRuntimeDescriptor(value) => value.bootstrap_source()
  }
  (
    source.kernel().sampling_runtime(),
    source.kernel().fused_runtime_source() is Some(_),
  )
}

///|
fn AdmittedRuntimeDescriptor::release_device(
  self : AdmittedRuntimeDescriptor,
) -> (Int, Int, Int) {
  let (ordinal, target) = match self {
    LegacyRuntimeDescriptor(value) => (value.device_ordinal(), value.target())
    I8RuntimeDescriptor(value) => (value.device_ordinal(), value.target())
    Fp8RuntimeDescriptor(value) => (value.device_ordinal(), value.target())
    TensorParallelRuntimeDescriptor(value) => {
      let topology = value.topology()
      (topology.rank(0).unwrap().ordinal(), topology.target())
    }
  }
  (ordinal, target.compute_major(), target.compute_minor())
}

///|
fn AdmittedRuntimeInstance::release_policy_sha256(
  self : AdmittedRuntimeInstance,
) -> String {
  match self {
    LegacyRuntimeInstance(value) => value.policy_digest().as_hex()
    I8RuntimeInstance(value) => value.policy_digest().as_hex()
    Fp8RuntimeInstance(value) => value.policy_digest().as_hex()
    TensorParallelRuntimeInstance(value) => value.policy_digest().as_hex()
  }
}

///|
fn AdmittedRuntimeInstance::release_tokenizer_sha256(
  self : AdmittedRuntimeInstance,
) -> String {
  match self {
    LegacyRuntimeInstance(value) => value.tokenizer_digest().as_string()
    I8RuntimeInstance(value) => value.tokenizer_digest().as_string()
    Fp8RuntimeInstance(value) => value.tokenizer_digest().as_string()
    TensorParallelRuntimeInstance(value) => value.tokenizer_digest().as_string()
  }
}

///|
/// Pre-resource parent identity for a future live tensor-parallel group.
fn AdmittedRuntimeInstance::release_group_template_sha256(
  self : AdmittedRuntimeInstance,
  executable : @worker_executable_file.WorkerExecutableAdmission,
) -> String? raise RuntimeInstanceError {
  match self {
    TensorParallelRuntimeInstance(instance) => {
      let runtime = instance.runtime_descriptor()
      let inputs = runtime.planning_inputs(
        instance.rank_group_process_limits() catch {
          _ => raise fail(InstanceAdmission)
        },
        executable.activation_path(),
        executable.digest().as_hex(),
      ) catch {
        _ => raise fail(InstanceAdmission)
      }
      Some(inputs.template_digest().as_sha256())
    }
    _ => None
  }
}

///|
fn verify_release_worker(
  launch : @launch_file.LaunchFileAdmission,
  limits : @worker_executable_file.WorkerExecutableFileLimits,
) -> @worker_executable_file.WorkerExecutableAdmission raise RuntimeInstanceError {
  let digest = @worker_executable_file.WorkerExecutableDigest::from_sha256(
    launch.worker_executable_sha256(),
  ) catch {
    _ => raise fail(WorkerExecutableAdmission)
  }
  @worker_executable_file.verify(
    launch.worker_executable_path(),
    digest,
    limits,
  ) catch {
    _ => raise fail(WorkerExecutableAdmission)
  }
}

///|
fn encode_release_evidence(
  runtime_recipe : String,
  launch_sha256 : String,
  descriptor_sha256 : String,
  policy_sha256 : String,
  tokenizer_sha256 : String,
  worker_sha256 : String,
  model_content_sha256 : String,
  model_plan_sha256 : String,
  bootstrap_sha256 : String,
  bootstrap_source_sha256 : String,
  device_ordinal : Int,
  compute_major : Int,
  compute_minor : Int,
  tensor_parallel_group_template_sha256? : String? = None,
) -> (String, String) {
  let tensor_parallel_group_scope = match
    tensor_parallel_group_template_sha256 {
    Some(digest) => "tensor_parallel_group_template_sha256=\{digest}\n"
    None => ""
  }
  let evidence = "schema=lunaflux-release-preflight.v1\n" +
    "runtime_recipe=" +
    runtime_recipe +
    "\nlaunch_sha256=" +
    launch_sha256 +
    "\nruntime_descriptor_sha256=" +
    descriptor_sha256 +
    "\ninstance_policy_sha256=" +
    policy_sha256 +
    "\ntokenizer_sha256=" +
    tokenizer_sha256 +
    "\nworker_executable_sha256=" +
    worker_sha256 +
    "\nmodel_content_sha256=" +
    model_content_sha256 +
    "\nmodel_plan_sha256=" +
    model_plan_sha256 +
    "\nbootstrap_sha256=" +
    bootstrap_sha256 +
    "\nbootstrap_source_sha256=" +
    bootstrap_source_sha256 +
    "\n" +
    tensor_parallel_group_scope +
    "device_ordinal=" +
    device_ordinal.to_string() +
    "\ncompute_major=" +
    compute_major.to_string() +
    "\ncompute_minor=" +
    compute_minor.to_string() +
    "\nsemantic_join=1\nfilesystem_authority_closed=1\n" +
    "device_opened=0\ncompiler_jit_authority=0\n"
  (
    evidence,
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))),
  )
}

///|
fn build_release_admission(
  admitted : AdmittedReleaseInputs,
  executable : @worker_executable_file.WorkerExecutableAdmission,
) -> RuntimeReleaseAdmission raise RuntimeInstanceError {
  let runtime_recipe = runtime_recipe_name(admitted.launch.runtime_recipe())
  let launch_sha256 = admitted.launch.digest().as_hex()
  let descriptor_sha256 = admitted.runtime.descriptor_sha256()
  let policy_sha256 = admitted.instance.release_policy_sha256()
  let tokenizer_sha256 = admitted.instance.release_tokenizer_sha256()
  let worker_sha256 = executable.digest().as_hex()
  let (model_content_sha256, model_plan_sha256) = admitted.runtime.release_model_identity()
  let (bootstrap_sha256, bootstrap_source_sha256) = admitted.runtime.release_bootstrap()
  let (device_ordinal, compute_major, compute_minor) = admitted.runtime.release_device()
  let (sampling_runtime, has_fused_runtime) = admitted.runtime.release_execution_extensions()
  let group_template = admitted.instance.release_group_template_sha256(
    executable,
  )
  let (evidence, evidence_sha256) = encode_release_evidence(
    runtime_recipe,
    launch_sha256,
    descriptor_sha256,
    policy_sha256,
    tokenizer_sha256,
    worker_sha256,
    model_content_sha256,
    model_plan_sha256,
    bootstrap_sha256,
    bootstrap_source_sha256,
    device_ordinal,
    compute_major,
    compute_minor,
    tensor_parallel_group_template_sha256=group_template,
  )
  {
    runtime_recipe,
    model_content_sha256,
    model_plan_sha256,
    tokenizer_sha256,
    worker_executable_sha256: worker_sha256,
    sampling_runtime,
    has_fused_runtime,
    evidence,
    evidence_sha256,
  }
}

///|
fn RuntimeReleaseAdmission::matches_spawned_execution_extensions(
  self : RuntimeReleaseAdmission,
  expect_host_sampling : Bool,
  require_fused_runtime : Bool,
) -> Bool {
  let sampling_matches = if expect_host_sampling {
    self.sampling_runtime == HostSamplingRuntime
  } else {
    self.sampling_runtime == EmbeddedCudaGreedyV1
  }
  sampling_matches && (!require_fused_runtime || self.has_fused_runtime)
}

///|
fn release_preflight_with_cleanup(
  primary : RuntimeInstanceError,
) -> RuntimeInstanceError {
  match primary {
    RuntimeInstanceFailed(stage) =>
      RuntimeInstanceFailedAndCleanup(stage, Cleanup)
    RuntimeInstanceFailedAndCleanup(stage, _) =>
      RuntimeInstanceFailedAndCleanup(stage, Cleanup)
  }
}

///|
/// Authenticates one deployment release through the same recipe-specific
/// descriptor and instance joins as live startup, but deliberately stops before
/// device probing, process creation, socket binding, or mutable owner creation.
/// All opened roots are closed before root-free evidence is returned.
pub fn preflight_release(
  deployment_argument : StringView,
  limits : RuntimeInstanceFileLimits,
) -> RuntimeReleaseAdmission raise RuntimeInstanceError {
  let admitted = admit_release_inputs(deployment_argument, limits)
  let executable_result : Result[
    @worker_executable_file.WorkerExecutableAdmission,
    RuntimeInstanceError,
  ] = Ok(verify_release_worker(admitted.launch, limits.executable)) catch {
    error => Err(error)
  }
  let executable = match executable_result {
    Ok(value) => value
    Err(error) => {
      close_model_kernel(admitted.model_root, admitted.kernel_root) catch {
        _ => raise release_preflight_with_cleanup(error)
      }
      raise error
    }
  }
  let evidence : Result[RuntimeReleaseAdmission, RuntimeInstanceError] = Ok(
    build_release_admission(admitted, executable),
  ) catch {
    error => Err(error)
  }
  let cleanup_failed = Ref(false)
  executable.close() catch {
    _ => cleanup_failed.val = true
  }
  close_model_kernel(admitted.model_root, admitted.kernel_root) catch {
    _ => cleanup_failed.val = true
  }
  match evidence {
    Err(error) if cleanup_failed.val =>
      raise release_preflight_with_cleanup(error)
    Err(error) => raise error
    Ok(_) if cleanup_failed.val => raise fail(Cleanup)
    Ok(value) => value
  }
}

///|
pub fn RuntimeReleaseAdmission::runtime_recipe(
  self : RuntimeReleaseAdmission,
) -> String {
  self.runtime_recipe
}

///|
pub fn RuntimeReleaseAdmission::model_content_sha256(
  self : RuntimeReleaseAdmission,
) -> String {
  self.model_content_sha256
}

///|
pub fn RuntimeReleaseAdmission::model_plan_sha256(
  self : RuntimeReleaseAdmission,
) -> String {
  self.model_plan_sha256
}

///|
/// Returns the independently authenticated tokenizer artifact digest.
pub fn RuntimeReleaseAdmission::tokenizer_sha256(
  self : RuntimeReleaseAdmission,
) -> String {
  self.tokenizer_sha256
}

///|
/// Returns the independently authenticated spawned child executable digest.
pub fn RuntimeReleaseAdmission::worker_executable_sha256(
  self : RuntimeReleaseAdmission,
) -> String {
  self.worker_executable_sha256
}

///|
pub fn RuntimeReleaseAdmission::evidence(
  self : RuntimeReleaseAdmission,
) -> String {
  self.evidence
}

///|
pub fn RuntimeReleaseAdmission::evidence_sha256(
  self : RuntimeReleaseAdmission,
) -> String {
  self.evidence_sha256
}
