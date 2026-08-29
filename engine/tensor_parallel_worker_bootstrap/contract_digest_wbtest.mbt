///|
async test "device-plan digest binds canonical generic tensor order" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-worker-bootstrap-digest")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = tp_bootstrap_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let model = tp_bootstrap_model(file)
    let model_plan = @llama.build_paged(
      model,
      batch=@llama.LlamaPagedBatchEnvelope::new(2),
    )
    let topology = tp_bootstrap_topology()
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspections = tp_bootstrap_inspections(root, model, model_plan)
    let device_plans = tp_bootstrap_device_plans(
      topology, model_plan, inspections,
    )
    let device_plan = device_plans[0]
    let canonical : BootstrapDigestEncoder = { bytes: [] }
    canonical.append_string("generic-tensor-order-test.v1")
    append_tensor_table(canonical, device_plan)
    let reversed : BootstrapDigestEncoder = { bytes: [] }
    reversed.append_string("generic-tensor-order-test.v1")
    reversed.append_int(device_plan.tensor_count())
    for offset in 0..<device_plan.tensor_count() {
      let index = device_plan.tensor_count() - 1 - offset
      let reference = @plan.TensorRef::new(index)
      guard device_plan.tensor(reference) is Some(matrix) else {
        fail("generic tensor table is incomplete")
      }
      append_matrix(reversed, matrix)
    }
    assert_true(device_plan.tensor_count() > 1)
    assert_true(canonical.finish() != reversed.finish())
    assert_true(
      digest_device_plan(device_plans[0]) != digest_device_plan(device_plans[1]),
    )
  }
}
