///|
test "ordered decoder shares only dead sparse scratch and budgets it once" {
  let slice = GlmDecoderSlice::new(
    @glm53_spec.Glm53FlashSpec::official(),
    first_layer=0,
    end_layer=45,
    maximum_rows=1,
    maximum_sequences=1,
    cache_slots=1,
    maximum_history=64,
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=10000000L,
    maximum_layer_workspace_bytes=20000000L,
    maximum_layer_expert_bytes=5000000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
  )
  defer slice.close()
  let sparse = slice.layers.filter(layer => layer is Sparse(_))
  assert_eq(sparse.length(), 11)
  let scratch = sparse[0].transient_scratch_bytes()
  assert_true(scratch > 0L)
  assert_eq(slice.scratch_plan.bytes(), scratch)
  assert_eq(slice.scratch_plan.saved_bytes(), scratch * 10L)
  let unshared = slice.layers.fold(init=0L, (n, layer) => {
    n + layer.workspace_bytes()
  })
  assert_eq(
    slice.workspace_bytes(),
    unshared + 2L * slice.residual_bytes - 10L * scratch,
  )
  assert_true(slice.scratch_buffers.is_empty())
}
