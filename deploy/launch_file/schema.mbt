///|
fn parse_luna_approval(
  root : Map[String, Json],
) -> LaunchLunaApproval? raise LaunchFileError {
  let object = require_object(root, "luna_approval", Root)
  let mode = require_string(object, "mode", LunaApproval)
  match mode {
    "none" => {
      validate_fields(object, LunaApproval, ["mode"])
      None
    }
    "external_ed25519_v1" => {
      validate_fields(object, LunaApproval, [
        "mode", "capability_manifest_sha256", "key_identity_sha256", "signature_sha256",
        "approved_source_sha256",
      ])
      Some({
        capability_manifest_sha256: require_sha256(
          object,
          "capability_manifest_sha256",
          LunaApproval,
        ),
        key_identity_sha256: require_sha256(
          object,
          "key_identity_sha256",
          LunaApproval,
        ),
        signature_sha256: require_sha256(
          object,
          "signature_sha256",
          LunaApproval,
        ),
        approved_source_sha256: require_sha256(
          object,
          "approved_source_sha256",
          LunaApproval,
        ),
      })
    }
    _ => raise InvalidField(LunaApproval)
  }
}

///|
fn parse_launch(
  bytes : BytesView,
  digest : LaunchFileDigest,
  limits : LaunchFileLimits,
) -> LaunchFileAdmission raise LaunchFileError {
  let text = @utf8.decode(bytes) catch { _ => raise InvalidUtf8 }
  @json_guard.reject_duplicate_object_keys(
    text,
    max_depth=limits.maximum_json_depth(),
  ) catch {
    DepthLimitExceeded(_) => raise JsonDepthExceeded
    DuplicateObjectKey(..) => raise DuplicateField
    _ => raise MalformedJson
  }
  let value = @json.parse(text) catch { _ => raise MalformedJson }
  guard value is Object(root) else { raise MalformedJson }
  let schema = require_string(root, "schema", Root)
  let runtime_recipe = match schema {
    "lunaflux.launch.v1" => {
      validate_fields(root, Root, [
        "schema", "model_root", "kernel_root", "policy_root", "runtime_descriptor",
        "instance_policy", "worker_executable", "luna_approval",
      ])
      DenseLlamaPagedAotV5
    }
    "lunaflux.launch.v2" => {
      validate_fields(root, Root, [
        "schema", "runtime_recipe", "model_root", "kernel_root", "policy_root", "runtime_descriptor",
        "instance_policy", "worker_executable", "luna_approval",
      ])
      match require_string(root, "runtime_recipe", Root) {
        "dense_llama_paged_aot_v5" => DenseLlamaPagedAotV5
        "dense_llama_i8_paged_aot_v6" => DenseLlamaI8PagedAotV6
        "dense_mistral_bf16_paged_aot_v7" => DenseMistralBf16PagedAotV7
        _ => raise InvalidField(Root)
      }
    }
    "lunaflux.launch.v3" => {
      validate_fields(root, Root, [
        "schema", "runtime_recipe", "model_root", "kernel_root", "policy_root", "runtime_descriptor",
        "instance_policy", "worker_executable", "luna_approval",
      ])
      if require_string(root, "runtime_recipe", Root) !=
        "dense_llama_tensor_parallel_paged_aot_approved_v8" {
        raise InvalidField(Root)
      }
      DenseLlamaTensorParallelPagedAotApprovedV8
    }
    "lunaflux.launch.v4" => {
      validate_fields(root, Root, [
        "schema", "runtime_recipe", "model_root", "kernel_root", "policy_root", "runtime_descriptor",
        "instance_policy", "worker_executable", "luna_approval",
      ])
      if require_string(root, "runtime_recipe", Root) !=
        "dense_llama_fp8_reusable_paged_aot_approved_v9" {
        raise InvalidField(Root)
      }
      DenseLlamaFp8ReusablePagedAotApprovedV9
    }
    "lunaflux.launch.v5" => {
      validate_fields(root, Root, [
        "schema", "runtime_recipe", "model_root", "kernel_root", "policy_root", "runtime_descriptor",
        "instance_policy", "worker_executable", "luna_approval",
      ])
      if require_string(root, "runtime_recipe", Root) !=
        "dense_qwen3_bf16_paged_aot_v12" {
        raise InvalidField(Root)
      }
      DenseQwen3Bf16PagedAotV12
    }
    _ => raise InvalidField(Root)
  }
  let model_root = require_string(root, "model_root", Root)
  let kernel_root = require_string(root, "kernel_root", Root)
  let policy_root = require_string(root, "policy_root", Root)
  validate_absolute_label(model_root)
  validate_absolute_label(kernel_root)
  validate_absolute_label(policy_root)
  let descriptor = require_object(root, "runtime_descriptor", Root)
  validate_fields(descriptor, RuntimeDescriptor, ["locator", "sha256"])
  let descriptor_locator = require_string(
    descriptor,
    "locator",
    RuntimeDescriptor,
  )
  validate_locator(descriptor_locator, RuntimeDescriptor)
  let policy = require_object(root, "instance_policy", Root)
  validate_fields(policy, InstancePolicy, ["locator", "sha256"])
  let policy_locator = require_string(policy, "locator", InstancePolicy)
  validate_locator(policy_locator, InstancePolicy)
  let worker = require_object(root, "worker_executable", Root)
  validate_fields(worker, WorkerExecutable, ["path", "sha256"])
  let worker_path = require_string(worker, "path", WorkerExecutable)
  validate_absolute_label(worker_path)
  let luna_approval = parse_luna_approval(root)
  if runtime_recipe == DenseLlamaI8PagedAotV6 && luna_approval is Some(_) {
    // Recipe v6 has no field that could carry this approval into the child;
    // rejecting here prevents an authenticated claim from being discarded.
    raise InvalidField(LunaApproval)
  }
  if runtime_recipe == DenseMistralBf16PagedAotV7 && luna_approval is Some(_) {
    raise InvalidField(LunaApproval)
  }
  if runtime_recipe == DenseQwen3Bf16PagedAotV12 && luna_approval is Some(_) {
    raise InvalidField(LunaApproval)
  }
  if runtime_recipe == DenseLlamaTensorParallelPagedAotApprovedV8 &&
    luna_approval is None {
    raise MissingField(LunaApproval)
  }
  if runtime_recipe == DenseLlamaFp8ReusablePagedAotApprovedV9 &&
    luna_approval is None {
    raise MissingField(LunaApproval)
  }
  {
    digest,
    runtime_recipe,
    model_root,
    kernel_root,
    policy_root,
    descriptor_locator,
    descriptor_sha256: require_sha256(descriptor, "sha256", RuntimeDescriptor),
    policy_locator,
    policy_sha256: require_sha256(policy, "sha256", InstancePolicy),
    worker_executable_path: worker_path,
    worker_executable_sha256: require_sha256(worker, "sha256", WorkerExecutable),
    luna_approval,
  }
}
