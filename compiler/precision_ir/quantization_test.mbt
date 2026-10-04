///|
test "precision quantization publishes global then scales then payload with explicit effects" {
  for
    (format, expected) in [
      (@format.Format::Bf16, [@precision_ir.QuantizationPass::EncodePayload]),
      (Int4, [ScaleAmax, EncodePayload]),
      (MxFp4, [ScaleAmax, EncodePayload]),
      (NvFp4, [GlobalAmax, ScaleAmax, EncodePayload]),
    ] {
    for
      (role, placement, effect) in [
        (
          @precision_ir.OperandRole::Activation,
          @precision_ir.Placement::QuantizeBeforeConsumer,
          @precision_ir.NumericEffect::StepWrite,
        ),
        (ValueCache, QuantizeOnCacheWrite, CacheCommit),
      ] {
      let operand = @precision_ir.plan_operand(
        role~,
        representation=@format.Representation::new(format~, rows=3, columns=35),
        placement~,
        supported_placements=[placement],
        accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
        memory_ceiling=10000L,
      )
      let plan = operand.quantization()
      assert_eq(plan.passes().length(), expected.length())
      for i, pass in expected {
        assert_eq(plan.passes()[i], pass)
      }
      assert_eq(plan.effect(), effect)
    }
  }
}
