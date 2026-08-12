///|
test "static device plan pins identity target regions activations and kernels" {
  let semantic = @llama.build(device_plan_test_model())
  let layout = device_plan_test_layout()
  let kernels = device_plan_catalog_for(semantic)
  let result = @device_plan.build(semantic, layout, kernels)
  assert_eq(result.identity(), semantic.identity())
  assert_eq(result.shape_constraints(), semantic.shape_constraints())
  assert_eq(result.target(), kernels.device_target())
  assert_eq(result.catalog_version(), kernels.catalog_version())
  assert_eq(result.weight_arena_bytes(), layout.arena_bytes())
  assert_eq(result.weight_alignment(), layout.alignment_bytes())
  assert_eq(result.weight_tensor_count(), layout.length())
  assert_eq(result.workspace_bytes(), 16L)
  assert_eq(result.workspace_alignment(), 16L)
  assert_eq(result.operations().length(), semantic.operations().length())
  assert_eq(result.final_output(), semantic.final_output())
  let mut external_count = 0
  let mut weight_count = 0
  let mut activation_count = 0
  for index, operation in result.operations() {
    let source = semantic.operations()[index]
    assert_eq(operation.id(), source.id())
    assert_eq(operation.kind(), source.kind())
    assert_eq(operation.shape(), source.shape())
    assert_eq(operation.outputs(), source.outputs())
    assert_eq(operation.binding(), kernels.bindings()[index])
    for input in operation.inputs() {
      match input {
        ExternalInput(TokenIds) => external_count += 1
        WeightRegion(region) => {
          weight_count += 1
          assert_eq(region.offset_bytes() % result.weight_alignment(), 0L)
          assert_true(region.end_offset_bytes() <= result.weight_arena_bytes())
        }
        ActivationValue(reference) => {
          activation_count += 1
          assert_true(reference.as_int() < operation.id().as_int())
        }
      }
    }
  }
  assert_eq(external_count, 1)
  assert_eq(weight_count, 12)
  assert_true(activation_count > 0)
}
