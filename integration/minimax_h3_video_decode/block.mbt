///|
pub struct Block {
  source : Bytes
  symbols : ReadOnlyArray[String]
  stem : AttentionStem
  scratch_bytes : Int64
  qk_offset : Int64
  attention_offset : Int64
  projection_offset : Int64
  residual_offset : Int64
  mlp_offset : Int64
  mlp_bytes : Int64
}

///|
pub fn Block::new(
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  prefix~ : String,
  max_scratch_bytes~ : Int64,
) -> Self raise {
  let stem = AttentionStem::new(
    temporal~,
    height~,
    width~,
    layer=0,
    prefix~,
    max_scratch_bytes~,
  )
  let tail = @video_latent_embedding_cuda_source.transformer_tail_source(
    rows=stem.rows,
    hidden=2048,
    heads=32,
    head_dimension=64,
    intermediate=8192,
    prefix~,
  )
  let qk_offset = align256(stem.scratch_bytes)
  let attention_offset = align256(qk_offset + stem.qkv_bytes)
  let projection_offset = align256(attention_offset + stem.hidden_bytes)
  let residual_offset = align256(projection_offset + stem.hidden_bytes)
  let mlp_offset = align256(residual_offset + stem.hidden_bytes * 2L)
  let mlp_bytes = stem.rows.to_int64() * 8192L * 2L
  let scratch_bytes = mlp_offset + mlp_bytes
  guard scratch_bytes <= max_scratch_bytes else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let symbols = stem.symbols[:].to_owned()
  for suffix in ["_attention", "_project", "_residual", "_gate_up", "_down"] {
    symbols.push(prefix + suffix)
  }
  {
    source: stem.source + tail,
    symbols: ReadOnlyArray::from_array(symbols),
    stem,
    scratch_bytes,
    qk_offset,
    attention_offset,
    projection_offset,
    residual_offset,
    mlp_offset,
    mlp_bytes,
  }
}

///|
fn align256(bytes : Int64) -> Int64 {
  (bytes + 255L) / 256L * 256L
}

///|
/// Borrow caller-owned, distinct scratch/input/output allocations. The layer
/// packing is fixed before submission; no checkpoint work occurs on dispatch.
pub fn Block::launches(
  self : Self,
  functions : ArrayView[@device.Function],
  packing : @minimax_h3_weights.VideoBlockWeightPacking,
  input : @device.Allocation,
  weights : @device.Allocation,
  scratch : @device.Allocation,
  output : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 9 && scratch.byte_count() >= self.scratch_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn region(
    a : @device.Allocation,
    o : Int64,
    n : Int64,
    align : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      a,
      offset=o,
      byte_count=n,
      alignment=align,
    )
  }
  fn weight(i : Int) -> @device.KernelArgument raise @device.DeviceError {
    let copy = packing.copies[i]
    region(weights, copy.destination_offset, copy.byte_count, 4L)
  }
  fn launch(
    i : Int,
    elements : Int64,
    block : Int,
    args : Array[@device.KernelArgument],
  ) -> @device.OrderedKernelLaunch raise @device.DeviceError {
    @device.OrderedKernelLaunch::new(
      functions[i],
      @device.KernelLaunchDimensions::new(
        grid_x=((elements - 1L) / block.to_int64() + 1L).to_int(),
        grid_y=1,
        grid_z=1,
        block_x=block,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      @device.KernelArguments::new(args),
    )
  }
  let hb = self.stem.hidden_bytes
  let norm = region(scratch, 0L, hb, 2L)
  let qkv = region(scratch, self.stem.qkv_offset, self.stem.qkv_bytes, 2L)
  let table = region(scratch, self.stem.table_offset, self.stem.table_bytes, 2L)
  let qk = region(scratch, self.qk_offset, self.stem.qkv_bytes, 2L)
  let attention = region(scratch, self.attention_offset, hb, 2L)
  let projection = region(scratch, self.projection_offset, hb, 2L)
  let residual = region(scratch, self.residual_offset, hb * 2L, 4L)
  let mlp = region(scratch, self.mlp_offset, self.mlp_bytes, 2L)
  let input = region(input, 0L, hb * 2L, 4L)
  let output = region(output, 0L, hb * 2L, 4L)
  [
    launch(0, self.stem.rows.to_int64(), 1, [input, weight(0), norm]),
    launch(1, self.stem.qkv_bytes / 2L, 256, [
      norm,
      weight(1),
      weight(2),
      weight(3),
      weight(4),
      weight(5),
      weight(6),
      qkv,
    ]),
    launch(2, self.stem.table_bytes / 2L, 256, [table]),
    launch(3, self.stem.rows.to_int64() * 96L, 1, [qkv, table, qk]),
    launch(4, self.stem.rows.to_int64() * 32L, 1, [qk, attention]),
    launch(5, hb / 2L, 256, [attention, weight(7), weight(8), projection]),
    launch(6, hb / 2L, 256, [input, projection, weight(9), residual]),
    launch(0, self.stem.rows.to_int64(), 1, [residual, weight(10), norm]),
    launch(7, self.mlp_bytes / 2L, 256, [norm, weight(11), weight(12), mlp]),
    launch(8, hb / 2L, 256, [mlp, weight(13), weight(14), projection]),
    launch(6, hb / 2L, 256, [residual, projection, weight(15), output]),
  ]
}
