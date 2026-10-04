///|
test "compact format lowering preserves the existing physical matrix program" {
  for selected in [false, true] {
    let parameters : ProjectionParameters = {
      family: if selected {
        LanguageModelHead
      } else {
        DenseProjection
      },
      input_width: 128,
      output_width: 64,
      query_width: 0,
      intermediate_width: 0,
    }
    let runtime = projection_build_runtime_dispatch(
      parameters,
      32,
      projection_fixture_target(),
      [],
    )
    let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch=runtime.dispatch_plan(32),
      maximum_rows=32,
      maximum_tokens=32,
      demand=if selected { QueryRowEnds } else { AllTokenRows },
      single_row_numerics=AllowDeterministicDotTree,
    )
    let physical = projection_refine_physical(
      @luna_projection_tile_compiler.compile_projection_tile(
        projection_bind_tile_weights(request, parameters),
      ),
    )
    let matrix = physical.matrix.unwrap()
    let region = matrix.operands.region(1).unwrap()
    let (_, rows, columns) = region.geometry()
    let tile_elements = rows * columns * matrix.schedule.stage_count()
    let accuracy = @precision.Accuracy::new(absolute=0.2, relative=0.1)
    let activation = @precision.plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=Bf16,
        rows=32,
        columns=128,
      ),
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy~,
      memory_ceiling=100000L,
    )
    let output = @precision.plan_operand(
      role=Output,
      representation=@format.Representation::new(
        format=Bf16,
        rows=32,
        columns=64,
      ),
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy~,
      memory_ceiling=100000L,
    )
    for
      format in [
        @format.Format::Bf16,
        Fp16,
        F32,
        Fp8E4M3,
        Fp8E5M2,
        Int8,
        Int4,
        Fp4E2M1,
        NvFp4,
        MxFp4,
        MxFp8E4M3,
        MxFp8E5M2,
      ] {
      let weight = @precision.plan_operand(
        role=Weight,
        representation=@format.Representation::new(
          format~,
          rows=64,
          columns=128,
        ),
        placement=DecodeInOperandTile,
        supported_placements=[DecodeInOperandTile],
        accuracy~,
        memory_ceiling=1000000L,
        tile_elements~,
      )
      let numerical = @precision.LinearPrecision::new(
        activation~,
        weight~,
        output~,
        route=DecodeToBf16,
        accumulator=F32,
      )
      let lowered = precision_projection_source(
        physical,
        numerical,
        symbol="test_compact",
      )
      let source = @utf8.decode(lowered.source_bytes())
      assert_eq(lowered.physical(), physical)
      assert_true(source.contains("const unsigned char *weight"))
      assert_true(
        source.contains(
          "test_compact_weight_load(weight,weight_scale,weight_global",
        ),
      )
      assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
      assert_eq(source.contains("const int32_t *row_offsets"), selected)
      assert_false(source.contains("if (!(s>0.0f)"))
      assert_false(source.contains("weight[weight_base + inner]"))
    }
  }
}
