///|
async fn attention_fixture() -> (
  @operators.DecoderOperators,
  @minicpm_v47_attention.Compilation,
) {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  // Explicit conservative compiler profile. This is not a hardware observation.
  let result = @minicpm_v47_attention.compile(
    operators,
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    target_id=@strategy.AttentionTargetId::new(
      "sm121-unqualified-software-profile",
    ),
    capabilities=@strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=49152L,
      max_pipeline_stages=1,
      parallel_workgroup_target=1,
      supports_async_copy=false,
      matrix_shape=Some(
        @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
      ),
    ),
  ) catch {
    @aot.CudaAttentionTileAotError::InvalidCudaAttentionTileAot(reason) =>
      fail("attention source pipeline: \{Repr(reason)}")
    error => raise error
  }
  (operators, result)
}

///|
async test "selected 256-wide GQA emits existing paged prefill and decode sources" {
  let (_, result) = attention_fixture()
  assert_eq(result.abi().shape(), (16, 2, 256, 16))
  assert_eq(result.abi().layout(), (5120, 8192L, 256L))
  assert_eq(result.cache_bytes(), 8388608L)
  assert_eq(result.abi().limits(), (1, 32, 256))
  let prefill = @utf8.decode(result.prefill().source().bytes()[:])
  let decode = @utf8.decode(result.decode().source().bytes()[:])
  assert_true(prefill.contains("#define LF_HEAD_DIMENSION 256"))
  assert_true(decode.contains("#define LF_HEAD_DIMENSION 256"))
  assert_true(prefill.contains("key_cache"))
  assert_true(decode.contains("query_position"))
  assert_true(result.prefill().lowering().launch().3 <= 49152L)
  assert_true(result.decode().lowering().launch().3 <= 49152L)
}

///|
async test "full frame orders cache publication and keeps coordinate domains separate" {
  let (operators, compilation) = attention_fixture()
  let program = @minicpm_v47_attention.Program::new(operators, compilation)
  assert_eq(program.workspace_bytes(), 1740800L)
  assert_eq(program.regions().length(), 27)
  let mut end = 0L
  let mut scratch_count = 0
  for region in program.regions() {
    if region.workspace_offset is Some(offset) {
      assert_eq(offset % 16L, 0L)
      assert_true(offset >= end)
      end = offset + region.byte_count
      scratch_count += 1
    }
  }
  assert_eq(scratch_count, 10)
  assert_eq(end, program.workspace_bytes())
  assert_eq(program.region(KeyCache).workspace_offset, None)
  assert_eq(program.region(ValueCache).byte_count, 4194304L)
  assert_eq(program.region(RawQuery).byte_count, 524288L)
  assert_eq(program.region(PackedQkv).byte_count, 327680L)
  assert_eq(program.region(Input).byte_count, 131072L)
  let common = @utf8.decode(program.common_source_bytes()[:])
  for phase in [@minicpm_v47_attention.Phase::Prefill, Decode] {
    let stages = program.stages(phase)
    assert_eq(stages.length(), 13)
    assert_eq(stages[5].arguments(), [Counts, CanvasPositions, Cosine, Sine])
    assert_eq(stages[6].arguments(), [Counts, Query, Cosine, Sine, Query])
    assert_eq(stages[7].arguments(), [Counts, Key, Cosine, Sine, Key])
    assert_eq(stages[8].arguments(), [Counts, Query, Key, RawValue, PackedQkv])
    assert_eq(stages[9].arguments(), [
      Counts,
      CausalPositions,
      RowOffsets,
      SequenceLengths,
      PageOffsets,
      PhysicalPages,
      PackedQkv,
      KeyCache,
      ValueCache,
    ])
    assert_eq(stages[10].arguments(), [
      Counts,
      CausalPositions,
      RowOffsets,
      SequenceLengths,
      PageOffsets,
      PhysicalPages,
      PackedQkv,
      Context,
      KeyCache,
      ValueCache,
    ])
    assert_eq(stages[11].arguments(), [Counts, Context, RawQuery, GatedContext])
    assert_eq(stages[12].arguments(), [
      Counts,
      GatedContext,
      OutputWeight,
      Output,
    ])
    for index, stage in stages {
      assert_eq(stage.is_attention(), index == 10)
      if index != 10 {
        assert_true(common.contains("void " + stage.symbol() + "("))
      }
      assert_true(stage.launch().2 > 0)
      for slot in stage.arguments() {
        assert_true(program.region(slot).byte_count > 0L)
      }
      assert_eq(stage.arguments().contains(CanvasPositions), index == 5)
      assert_eq(
        stage.arguments().contains(CausalPositions),
        index == 9 || index == 10,
      )
    }
    assert_true(
      @utf8.decode(program.attention_source_bytes(phase)[:]).contains(
        "void " + stages[10].symbol() + "(",
      ),
    )
  }
  for index in 0..<13 {
    if index != 10 {
      assert_eq(
        program.stages(Prefill)[index].symbol(),
        program.stages(Decode)[index].symbol(),
      )
    }
  }
  // Pure close path is testable without claiming native device preparation.
  let frame = @minicpm_v47_attention.Frame::new(program)
  frame.close()
  frame.close()
  let closed = try frame.launches(Prefill) catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(closed)
}
