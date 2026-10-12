///|
test "nonlinear rounding choices are explicit and matrix logits stay unchanged" {
  for
    affine in [
      @precision.Float32AffineLaw::SeparateMultiplyAdd,
      FusedMultiplyAdd,
    ] {
    for
      sigmoid in [
        @precision.SigmoidEvaluationLaw::StableBranches,
        DirectReciprocal,
      ] {
      let source = @decoder_hyper_connection_cuda_source.staged_function_and_collapse_bf16_f32_source(
        stream_width=4,
        hidden_width=128,
        maximum_rows=3,
        control_epsilon=0.000001,
        normalization_epsilon=0.000001,
        function_symbol="nonlinear_probe",
        control_law=F32ProjectionThenNormalize,
        control_vector_affine_law=affine,
        control_sigmoid_law=sigmoid,
      )
      let text = @utf8.decode(source.source_bytes())
      assert_eq(
        text.contains("__fmaf_rn(mixes[p], scale[1], base[p])"),
        affine == FusedMultiplyAdd,
      )
      assert_eq(text.contains("if (x >= 0.0f)"), sigmoid == StableBranches)
      assert_true(
        text.contains(
          "__fadd_rn(__fmul_rn(mixes[control_index], scale[2]), base[control_index])",
        ),
      )
      assert_false(text.contains("__expf"))
      if affine == FusedMultiplyAdd || sigmoid == DirectReciprocal {
        assert_eq(
          source.numerical_contract(),
          RefinedControlFunctionBf16CollapseV1(
            control=F32ProjectionThenNormalize,
            square=OrderedFloat32,
            dot=OrderedFloat32,
            vector_affine=affine,
            sigmoid~,
            root=DivideRoundedSqrtFloat32,
          ),
        )
      }
    }
  }
}
