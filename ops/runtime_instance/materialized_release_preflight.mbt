///|
fn encode_materialized_release_evidence(
  admitted : AdmittedMaterializedReleaseInputs,
) -> (String, String) raise RuntimeInstanceError {
  let runtime_recipe = runtime_recipe_name(admitted.launch.runtime_recipe())
  let launch_sha256 = admitted.launch.digest().as_hex()
  let descriptor_sha256 = admitted.runtime.descriptor_sha256()
  let policy_sha256 = admitted.instance.release_policy_sha256()
  let tokenizer_sha256 = admitted.instance.release_tokenizer_sha256()
  let worker_sha256 = admitted.executable.digest().as_hex()
  let (model_content_sha256, model_plan_sha256) = admitted.runtime.release_model_identity()
  let (bootstrap_sha256, bootstrap_source_sha256) = admitted.runtime.release_bootstrap()
  let (device_ordinal, compute_major, compute_minor) = admitted.runtime.release_device()
  let group_template = match admitted.instance {
    TensorParallelRuntimeInstance(instance) => {
      let runtime = instance.runtime_descriptor()
      let inputs = runtime.planning_inputs(
        instance.rank_group_process_limits() catch {
          _ => raise fail(InstanceAdmission)
        },
        admitted.executable.activation_path(),
        admitted.executable.digest().as_hex(),
      ) catch {
        _ => raise fail(InstanceAdmission)
      }
      Some(inputs.template_digest().as_sha256())
    }
    _ => None
  }
  let tensor_parallel_group_scope = match group_template {
    Some(digest) => "tensor_parallel_group_template_sha256=\{digest}\n"
    None => ""
  }
  let evidence = "schema=lunaflux-release-materialization-preflight.v1\n" +
    "runtime_recipe=" +
    runtime_recipe +
    "\ndeployment_bundle_sha256=" +
    admitted.bundle_sha256 +
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
    "\nsource_target_binding=1\nsemantic_join=1\n" +
    "filesystem_authority_closed=1\ndevice_opened=0\n" +
    "compiler_jit_authority=0\n"
  (
    evidence,
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(evidence))),
  )
}

///|
fn build_materialized_release_admission(
  admitted : AdmittedMaterializedReleaseInputs,
) -> RuntimeReleaseAdmission raise RuntimeInstanceError {
  let (model_content_sha256, model_plan_sha256) = admitted.runtime.release_model_identity()
  let (evidence, evidence_sha256) = encode_materialized_release_evidence(
    admitted,
  )
  let (sampling_runtime, has_fused_runtime) = admitted.runtime.release_execution_extensions()
  {
    runtime_recipe: runtime_recipe_name(admitted.launch.runtime_recipe()),
    model_content_sha256,
    model_plan_sha256,
    tokenizer_sha256: admitted.instance.release_tokenizer_sha256(),
    worker_executable_sha256: admitted.executable.digest().as_hex(),
    sampling_runtime,
    has_fused_runtime,
    evidence,
    evidence_sha256,
  }
}

///|
/// Authenticates one exact no-overwrite deployment-bundle materialization.
///
/// Staged source roots are pinned through no-follow capabilities. Typed views
/// bind those bytes to the final absolute model/kernel/policy/executable labels
/// carried by the launch file, so descriptor/bootstrap reconstruction keeps
/// production identity without requiring privileged chroot or mount setup.
/// All source authority closes before root-free evidence is returned. This
/// path opens no device, process, listener, compiler, or JIT authority.
pub fn preflight_materialized_release(
  bundle_argument : StringView,
  limits : RuntimeInstanceFileLimits,
  bundle_limits : @deployment_bundle.DeploymentBundleLimits,
) -> RuntimeReleaseAdmission raise RuntimeInstanceError {
  let bundle = load_materialized_bundle(bundle_argument, bundle_limits)
  let admitted = admit_materialized_release_inputs(bundle, limits)
  let evidence = build_materialized_release_admission(admitted)
  close_three_roots(
    admitted.model_root,
    admitted.kernel_root,
    admitted.worker_root,
  )
  evidence
}
