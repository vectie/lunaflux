///|
async test "materialized descriptor reads pinned source and retains target labels" {
  let canonical = descriptor_fixture()
  with_descriptor_roots(canonical, (label, model_root, kernel_root) => {
    let target_model = "/var/lib/lunaflux/model"
    let target_kernel = "/opt/lunaflux/kernels"
    let model = try! model_root.materialization_view(label, target_model)
    let kernel = try! kernel_root.materialization_view(label, target_kernel)
    try
      @descriptor_file.load_materialized(
        model,
        kernel,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(ModelConfig) => ()
      _ => abort("mapped legacy descriptor failed before its pinned descendant")
    } noraise {
      _ => abort("descriptor without model files was admitted")
    }
    if model_root.is_closed() || kernel_root.is_closed() {
      abort("mapped legacy loader consumed caller roots")
    }
  })
}

///|
async test "materialized I8 descriptor never requires target namespace access" {
  let canonical = i8_public_descriptor_fixture()
  with_descriptor_roots(canonical, (label, model_root, kernel_root) => {
    let model = try! model_root.materialization_view(
      label, "/var/lib/lunaflux/model",
    )
    let kernel = try! kernel_root.materialization_view(
      label, "/opt/lunaflux/kernels",
    )
    try
      @descriptor_file.load_i8_v2_materialized(
        model,
        kernel,
        try! @approved_fs.ApprovedRelativeLocator::new("runtime.json"),
        i8_descriptor_digest(canonical),
        @descriptor_file.RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      )
    catch {
      @descriptor_file.AdmissionRejected(ModelConfig) => ()
      _ => abort("mapped I8 descriptor failed before its pinned descendant")
    } noraise {
      _ => abort("I8 descriptor without model files was admitted")
    }
    if model_root.is_closed() || kernel_root.is_closed() {
      abort("mapped I8 loader consumed caller roots")
    }
  })
}
