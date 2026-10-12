///|
test "control nonlinear laws reach all attention and FFN envelopes without changing ordinary sources" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let ordinary = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
  defer ordinary.close()
  for
    (affine, sigmoid) in [
      (
        @precision.Float32AffineLaw::SeparateMultiplyAdd,
        @precision.SigmoidEvaluationLaw::StableBranches,
      ),
      (FusedMultiplyAdd, DirectReciprocal),
    ] {
    let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
      spec,
      first_layer=0,
      end_layer=4,
      maximum_rows=3,
      maximum_context=256,
      maximum_layer_state_bytes=16777216L,
      maximum_layer_workspace_bytes=16777216L,
      maximum_layer_expert_bytes=8589934592L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
      control_vector_affine_law=affine,
      control_sigmoid_law=sigmoid,
    )
    defer selected.close()
    let source = selected.source_bytes()
    if affine == SeparateMultiplyAdd {
      assert_eq(source, ordinary.source_bytes())
    } else {
      let text = @utf8.decode(source)
      assert_eq(text.split("__fmaf_rn(mixes[p], scale[1], base[p])").count(), 9)
      assert_eq(
        text
        .split(
          "__fadd_rn(__fmul_rn(mixes[control_index], scale[2]), base[control_index])",
        )
        .count(),
        25,
      )
    }
    assert_eq(selected.expert_bank_bytes(), ordinary.expert_bank_bytes())
  }
}
