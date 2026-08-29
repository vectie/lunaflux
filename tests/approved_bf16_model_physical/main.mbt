///|
async fn main {
  match @env.args() {
    [_, "export", model_root, toolchain_sha256, major, minor, patch, output] => {
      require_compiled_sha256(toolchain_sha256, "toolchain digest is invalid")
      let compiler_major = parse_physical_compiler_component(major, false)
      let compiler_minor = parse_physical_compiler_component(minor, true)
      let compiler_patch = parse_physical_compiler_component(patch, true)
      let inputs = load_approved_bf16_inputs(model_root)
      let set = approved_candidate_set(
        inputs, toolchain_sha256, compiler_major, compiler_minor, compiler_patch,
      )
      let evidence = @luna_bf16_candidate_export.export_new(set, output)
      println(
        "outcome=approved-bf16-candidates-exported target=\{evidence.target()} " +
        "operations=\{evidence.operation_count()} " +
        "candidate_set_sha256=\{evidence.candidate_set_sha256()} " +
        "inventory_sha256=\{evidence.inventory_sha256()} " +
        "export_record_sha256=\{evidence.export_record_sha256()}",
      )
    }
    [
      _,
      "run",
      model_root,
      compiled_root,
      runtime_root,
      toolchain_manifest,
      toolchain_sha256,
      major,
      minor,
      patch,
    ] => {
      require_compiled_sha256(toolchain_sha256, "toolchain digest is invalid")
      let toolchain_bytes = @fs.read_file(toolchain_manifest).binary()
      if approved_sha256(toolchain_bytes) != toolchain_sha256 {
        raise InvalidInput("toolchain manifest digest mismatch")
      }
      let toolchain_text = @utf8.decode(toolchain_bytes)
      if require_compiled_value(toolchain_text, "schema") !=
        "lunaflux-approved-complete-cuda-toolchain-v1" {
        raise InvalidInput("toolchain manifest schema mismatch")
      }
      let driver_identity_sha256 = require_compiled_value(
        toolchain_text, "driver_identity_sha256",
      )
      require_compiled_sha256(
        driver_identity_sha256, "driver identity digest is invalid",
      )
      let compiler_major = parse_physical_compiler_component(major, false)
      let compiler_minor = parse_physical_compiler_component(minor, true)
      let compiler_patch = parse_physical_compiler_component(patch, true)
      let maximum_error = run_approved_bf16_physical(
        model_root, compiled_root, runtime_root, toolchain_sha256, driver_identity_sha256,
        compiler_major, compiler_minor, compiler_patch,
      )
      println(
        "outcome=approved-bf16-model-sm120-pass " +
        "revision=\{PINNED_REVISION} plan_executed=1 " +
        "selected_logits=11 max_abs_error=\{maximum_error} " +
        "greedy_tokens=1031,2185,688,2844 same_page_kv=pass " +
        "capture=required artifacts=authenticated weights=authenticated " +
        "resources=closed scope=tiny-approved-model-numerics",
      )
    }
    _ =>
      raise InvalidInput(
        "usage: approved_bf16_model_physical export MODEL_ROOT TOOLCHAIN_SHA256 COMPILER_MAJOR COMPILER_MINOR COMPILER_PATCH OUTPUT | run MODEL_ROOT COMPILED_ROOT RUNTIME_ROOT TOOLCHAIN_MANIFEST TOOLCHAIN_SHA256 COMPILER_MAJOR COMPILER_MINOR COMPILER_PATCH",
      )
  }
}
