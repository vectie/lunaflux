///|
fn fused_compiled_family_name(family : FusedParallelFamily) -> String {
  match family {
    QkvPositionedRopePagedKvWrite => "qkv-positioned-rope-paged-kvwrite"
    ResidualRmsNorm => "residual-rmsnorm"
  }
}

///|
fn fused_compiled_target_name(target : @catalog.DeviceTarget) -> String {
  "sm_" +
  target.compute_major().to_string() +
  target.compute_minor().to_string()
}

///|
fn fused_receipt_line_value(
  lines : ArrayView[String],
  index : Int32,
  key : String,
) -> String raise FusedParallelAotError {
  if index < 0 || index >= lines.length() {
    raise FusedParallelInvalid(CompileReceipt)
  }
  let prefix = key + "="
  let line = lines[index]
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise FusedParallelInvalid(CompileReceipt)
  }
  line[prefix.length():].to_owned()
}

///|
fn fused_require_recipe_abi(
  recipe : Bytes,
  abi : FusedParallelKernelAbi,
  dispatch_canary_per_token : UInt32,
  qualification_candidate : FusedParallelDigest?,
) -> Unit raise FusedParallelAotError {
  let document = @utf8.decode(recipe) catch {
    _ => raise FusedParallelInvalid(Recipe)
  }
  let expected = "abi=" + abi.canonical_name()
  let mut abi_count = 0
  let mut canary_count = 0
  let mut diagnostic_absent_count = 0
  let mut qualification_count = 0
  for line in document.split("\n") {
    if line.has_prefix("abi=") {
      abi_count += 1
      if line != expected {
        raise FusedParallelMismatch(KernelAbi)
      }
    }
    if line.has_prefix("dispatch_canary=") {
      canary_count += 1
      if line != "dispatch_canary=" + dispatch_canary_per_token.to_string() {
        raise FusedParallelMismatch(KernelAbi)
      }
    }
    if line.has_prefix("diagnostic_canary=") {
      diagnostic_absent_count += 1
      if line != "diagnostic_canary=absent" {
        raise FusedParallelMismatch(KernelAbi)
      }
    }
    if line.has_prefix("qualification_candidate_sha256=") {
      qualification_count += 1
      guard qualification_candidate is Some(parent) &&
        line == "qualification_candidate_sha256=" + parent.as_hex() else {
        raise FusedParallelMismatch(Digest)
      }
    }
  }
  match abi {
    ResidualRmsNormProductionAbiV2
    | QkvPositionedRopePagedKvWriteProductionAbiV2
    | QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4 =>
      if abi_count != 1 ||
        canary_count != 0 ||
        diagnostic_absent_count != 1 ||
        qualification_count != 1 ||
        dispatch_canary_per_token != 0U ||
        qualification_candidate is None {
        raise FusedParallelMismatch(KernelAbi)
      }
    _ =>
      if abi_count != 1 ||
        canary_count != 1 ||
        diagnostic_absent_count != 0 ||
        qualification_count != 0 ||
        dispatch_canary_per_token == 0U ||
        qualification_candidate is Some(_) {
        raise FusedParallelMismatch(KernelAbi)
      }
  }
}

///|
fn fused_compiled_binding_record(
  family : FusedParallelFamily,
  candidate_digest : FusedParallelDigest,
  source_digest : FusedParallelDigest,
  recipe_digest : FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  driver_identity_sha256 : String,
  target : @catalog.DeviceTarget,
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  abi : FusedParallelKernelAbi,
  dispatch_canary_per_token : UInt32,
  qualification_candidate : FusedParallelDigest?,
  module_digest : FusedParallelDigest,
  receipt_digest : FusedParallelDigest,
) -> Bytes {
  let identity = compiler.identity()
  let qualification_record = match qualification_candidate {
    Some(parent) => "qualification_candidate_sha256=" + parent.as_hex() + "\n"
    None => "qualification_candidate_sha256=absent\n"
  }
  @utf8.encode(
    "schema=lunaflux-fused-parallel-compiled-candidate-binding.v1\n" +
    "family=" +
    fused_compiled_family_name(family) +
    "\n" +
    "candidate_sha256=" +
    candidate_digest.as_hex() +
    "\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    recipe_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    driver_identity_sha256 +
    "\n" +
    "compiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    compiler.fused_multiply_add().to_string() +
    "\n" +
    "reassociate=" +
    compiler.reassociate().to_string() +
    "\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "target=" +
    fused_compiled_target_name(target) +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    "," +
    dimensions.grid_y().to_string() +
    "," +
    dimensions.grid_z().to_string() +
    "\n" +
    "block=" +
    dimensions.block_x().to_string() +
    "," +
    dimensions.block_y().to_string() +
    "," +
    dimensions.block_z().to_string() +
    "\n" +
    "shared_memory_bytes=" +
    dimensions.shared_memory_bytes().to_string() +
    "\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "abi=" +
    abi.canonical_name() +
    "\n" +
    "dispatch_canary_per_token=" +
    dispatch_canary_per_token.to_string() +
    "\n" +
    qualification_record +
    "cubin_sha256=" +
    module_digest.as_hex() +
    "\n" +
    "compile_receipt_sha256=" +
    receipt_digest.as_hex() +
    "\n" +
    "deterministic=1\n" +
    "candidate_only=true\n" +
    "compiler_authority=absent\n" +
    "deployment_approval=absent\n" +
    "promotion_authority=absent\n" +
    "manifest_bindable=false\n",
  )
}
