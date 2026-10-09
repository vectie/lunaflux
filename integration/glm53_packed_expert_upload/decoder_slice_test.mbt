///|
fn decoder_slice_fixture(
  first : Int,
  end : Int,
  sequences? : Int = 1,
) -> @glm53_packed_expert_upload.GlmDecoderSlice raise {
  @glm53_packed_expert_upload.GlmDecoderSlice::new(
    @glm53_spec.Glm53FlashSpec::official(),
    first_layer=first,
    end_layer=end,
    maximum_rows=1,
    maximum_sequences=sequences,
    cache_slots=sequences,
    maximum_history=64,
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=10000000L,
    maximum_layer_workspace_bytes=20000000L,
    maximum_layer_expert_bytes=5000000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
  )
}

///|
test "decoder slice resolves DSA and KDA MoE stages from the official schedule" {
  let slice = decoder_slice_fixture(3, 5)
  assert_eq(slice.layer_range(), (3, 5))
  let source = @utf8.decode(slice.source_bytes())
  assert_true(source.contains("glm_sparse_3_output("))
  assert_true(source.contains("glm_recurrent_4_15("))
  assert_true(source.contains("glm_routed_3_"))
  assert_true(source.contains("glm_routed_4_"))
  assert_eq(slice.launch_count(), 0)
  assert_true(slice.workspace_bytes() > 32768L)
  slice.close()
  slice.close()
}

///|
test "the whole decoder is expressible without allocating device weights" {
  let slice = decoder_slice_fixture(0, 45)
  assert_eq(slice.layer_range(), (0, 45))
  assert_true(slice.workspace_bytes() < 300000000L)
  let source = @utf8.decode(slice.source_bytes())
  assert_true(source.contains("glm_dense_ffn_0_"))
  assert_true(source.contains("glm_sparse_43_output("))
  assert_true(source.contains("glm_recurrent_44_15("))
  slice.close()
}

///|
test "sparse decoder slices do not pretend contiguous history is batched state" {
  try decoder_slice_fixture(3, 5, sequences=2) catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    error => raise error
  } noraise {
    _ => fail("multiple requests share one DSA history")
  }
}
