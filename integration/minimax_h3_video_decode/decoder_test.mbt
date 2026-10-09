///|
test "full tile decoder covers thirty six blocks and biased F32 final head" {
  let plan = @minimax_h3_video_decode.TileDecoder::new(
    temporal=7,
    height=2,
    width=3,
    prefix="tile_decoder",
    max_workspace_bytes=100000000L,
  )
  assert_eq(plan.block_weights.length(), 36)
  for i, weights in plan.block_weights {
    assert_eq(weights.layer, i)
    assert_eq(weights.copies.length(), 16)
    assert_eq(weights.byte_count, 268574720L)
    assert_eq(
      weights.copies[15].tensor_name,
      "decoder.transformer_blocks.\{i}.scale2",
    )
  }
  assert_eq(plan.head.weights.byte_count, 25194496L)
  assert_eq(plan.head.output_bytes, 7L * 2L * 3L * 3072L * 4L)
  let source = @utf8.decode(plan.source)
  assert_true(source.contains("tile_decoder_block_attention"))
  assert_true(source.contains("float maximum=-INFINITY"))
  assert_true(source.contains("float mean=0.0f"))
  assert_true(source.contains("variance"))
  assert_true(source.contains("tile_decoder_block_gate_up"))
  assert_true(source.contains("tile_decoder_last_head_unpack"))
  let owner = @minimax_h3_video_decode.TileDecoderProgram::new(plan)
  try owner.with_pixels(_ => ()) catch {
    _ => ()
  } noraise {
    _ => fail("unprepared output")
  }
  owner.close()
  owner.close()
}

///|
test "released spatial tile starts distribute overlap and cover exact canvas" {
  let y = @minimax_h3_video_decode.TileAxis::new(
    768,
    tile_size=256,
    minimum_overlap=64,
  )
  assert_eq(y.starts[:].to_owned(), [0, 160, 336, 512])
  assert_eq(y.overlaps[:].to_owned(), [96, 80, 80])
  let x = @minimax_h3_video_decode.TileAxis::new(
    1344,
    tile_size=256,
    minimum_overlap=64,
  )
  assert_eq(x.starts[:].to_owned(), [0, 176, 352, 528, 704, 896, 1088])
  assert_eq(x.starts[x.starts.length() - 1] + x.length, 1344)
  try
    @minimax_h3_video_decode.TileAxis::new(
      2147483632,
      tile_size=16,
      minimum_overlap=0,
      max_tiles=8,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unbounded tile allocation")
  }
}

///|
test "released temporal cadence and causal front trim" {
  let full = @minimax_h3_video_decode.TemporalTiles::new(37, output_frames=124)
  assert_eq(full.chunks, 7)
  assert_eq(full.pad_tokens, 0)
  assert_eq(full.available_frames, 124)
  assert_eq(full.trim_front, 0)
  let padded = @minimax_h3_video_decode.TemporalTiles::new(
    36,
    output_frames=120,
  )
  assert_eq(padded.pad_tokens, 1)
  assert_eq(padded.available_frames, 120)
  let trimmed = @minimax_h3_video_decode.TemporalTiles::new(
    37,
    output_frames=100,
  )
  assert_eq(trimmed.trim_front, 24)
  try @minimax_h3_video_decode.TemporalTiles::new(37, output_frames=125) catch {
    _ => ()
  } noraise {
    _ => fail("extra frames fabricated")
  }
}

///|
test "complete tiled decoder budgets all live storage and exposes shape" {
  let plan = @minimax_h3_video_decode.TiledDecoder::new(
    latent_frames=7,
    latent_height=4,
    latent_width=5,
    output_frames=22,
    tile_size=48,
    minimum_overlap=16,
    prefix="request",
    max_jobs=64,
    max_workspace_bytes=100000000L,
  )
  assert_eq(plan.temporal.chunks, 1)
  assert_eq(plan.tile_count, 4)
  assert_eq(plan.output_height, 64)
  assert_eq(plan.output_width, 80)
  assert_eq(plan.output_bytes, 3L * 22L * 64L * 80L * 4L)
  assert_eq(plan.symbols.length(), 7)
  assert_eq(
    plan.workspace_bytes,
    plan.tile.workspace_bytes +
    plan.gather_bytes +
    plan.tile_store_bytes +
    2L * plan.clip_bytes +
    plan.output_bytes,
  )
  let source = @utf8.decode(plan.source)
  assert_true(source.contains("request_gather_0_1_1"))
  assert_true(source.contains("request_publish_0"))
  assert_true(source.contains("if(iy>0)"))
  assert_true(source.contains("if(ix>0)"))
  let program = @minimax_h3_video_decode.TiledDecoderProgram::new(plan)
  assert_eq(program.plan().output_bytes, plan.output_bytes)
  try program.with_pixels(_ => ()) catch {
    _ => ()
  } noraise {
    _ => fail("unprepared pixels")
  }
  program.close()
  try
    @minimax_h3_video_decode.TiledDecoder::new(
      latent_frames=7,
      latent_height=4,
      latent_width=5,
      output_frames=22,
      tile_size=48,
      minimum_overlap=16,
      prefix="request",
      max_jobs=3,
      max_workspace_bytes=100000000L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("job budget ignored")
  }
  try
    @minimax_h3_video_decode.TiledDecoder::new(
      latent_frames=7,
      latent_height=4,
      latent_width=5,
      output_frames=22,
      tile_size=48,
      minimum_overlap=16,
      prefix="request",
      max_jobs=64,
      max_workspace_bytes=plan.workspace_bytes - 1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("workspace budget ignored")
  }
}
