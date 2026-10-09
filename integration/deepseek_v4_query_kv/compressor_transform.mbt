///|
pub fn indexer_cache_transform_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.RotaryHadamardFp4Precision raise {
  guard layer >= 0 &&
    layer < spec.layer_count() &&
    spec.compression_ratio(layer) == Some(4) &&
    maximum_rows > 0 &&
    maximum_rows <= 65536 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.RotaryHadamardFp4Precision::new(
    head_count=1,
    head_width=spec.index_head_dimension(),
    rotary_width=spec.rope_head_dimension(),
    maximum_rows=(maximum_rows + 3) / 4,
    maximum_positions=spec.max_position_embeddings(),
    frequency=rotary_frequency_policy(spec, layer~),
    simulation_block_width=32,
    maximum_workspace_bytes~,
  )
}

///|
priv enum CompressorTransform {
  Attention(@rotary_frame.RotaryCacheFrame)
  Indexer(@indexer_frame.RotaryHadamardFp4Frame)
}

///|
fn CompressorTransform::workspace_bytes(self : Self) -> Int64 {
  match self {
    Attention(frame) => frame.workspace_bytes()
    Indexer(frame) => frame.workspace_bytes()
  }
}

///|
fn CompressorTransform::source_bytes(self : Self) -> Bytes {
  match self {
    Attention(frame) => frame.source_bytes()
    Indexer(frame) => frame.source_bytes()
  }
}

///|
fn CompressorTransform::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  pool : DeepSeekCompressorPool,
) -> Unit raise {
  match self {
    Attention(frame) =>
      frame.prepare(
        context,
        aot,
        counts=pool.output_counts(),
        positions=pool.output_positions(),
        input=pool.output(),
        maximum_workspace_bytes=frame.workspace_bytes(),
      )
    Indexer(frame) =>
      frame.prepare(
        context,
        aot,
        counts=pool.output_counts(),
        positions=pool.output_positions(),
        input=pool.output(),
        maximum_workspace_bytes=frame.workspace_bytes(),
      )
  }
}

///|
fn CompressorTransform::output(self : Self) -> @device.Allocation raise {
  match self {
    Attention(frame) => frame.output()
    Indexer(frame) => frame.output()
  }
}

///|
fn CompressorTransform::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  match self {
    Attention(frame) => frame.launches()
    Indexer(frame) => frame.launches()
  }
}

///|
fn CompressorTransform::close(self : Self) -> Unit raise {
  match self {
    Attention(frame) => frame.close()
    Indexer(frame) => frame.close()
  }
}
