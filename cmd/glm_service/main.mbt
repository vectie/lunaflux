///|
async fn main {
  if @env.args() is [_] {
    inherited_worker() catch {
      error => {
        @stdio.stderr.write("GLM worker failed: \{Repr(error)}\n")
        @process.exit_failure()
      }
    }
    return
  }
  guard @env.args() is [_, mode, kernel_root, manifest] else {
    fail("export|rank1|serve KERNEL_ROOT MANIFEST")
  }
  let kernels = @approved_fs.ApprovedRoot::open_absolute(kernel_root)
  defer kernels.close()
  let plan = parse_plan(read_bounded(kernels, manifest, 65536L))
  guard plan.kernel_root == kernel_root else { fail("kernel root mismatch") }
  let root = @approved_fs.ApprovedRoot::open_absolute(plan.model_root)
  defer root.close()
  if mode == "rank1" {
    run_rank(plan, root, kernels, None, None)
    return
  }
  if mode == "export" {
    let spec = inspect_spec(root)
    let checkpoint = inspect_checkpoint(root, kernels, plan)
    for ordinal in 0..<2 {
      let stage = plan.stage(spec, ordinal)
      defer stage.close()
      let required = stage.required_device_bytes(checkpoint)
      guard required <= plan.budget - plan.reserve else {
        fail("GLM exceeds memory envelope before export")
      }
      @fs.write_file(
        kernel_root + "/rank\{ordinal}.cu",
        stage.source_bytes(),
        create_mode=CreateNew,
      )
      println(
        "export rank=\{ordinal} total=\{required+plan.reserve} executable=\{plan.executable}",
      )
    }
    let sampling = @sampling.FullVocabularySamplingProgram::new(
      values=spec.vocabulary_size(),
      maximum_rows=plan.slots,
      maximum_workspace_bytes=plan.reserve / 8L,
    ).unwrap()
    @fs.write_file(
      kernel_root + "/sampling.cu",
      @sampling_source.source(sampling),
      create_mode=CreateNew,
    )
    return
  }
  if mode == "serve" {
    serve(plan, manifest, root, kernels)
    return
  }
  fail("unknown GLM service operation")
}
