///|
/// One tile, one batch. Q/K/V output is planar BF16 [3,rows,32,64].
pub struct AttentionStem {
  source : Bytes
  symbols : ReadOnlyArray[String]
  rows : Int
  hidden_bytes : Int64
  qkv_bytes : Int64
  table_bytes : Int64
  qkv_offset : Int64
  table_offset : Int64
  scratch_bytes : Int64
  weights : @minimax_h3_weights.VideoAttentionWeightPacking
}

///|
pub fn AttentionStem::new(
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  layer~ : Int,
  prefix~ : String,
  max_scratch_bytes~ : Int64,
) -> Self raise {
  let source = @video_latent_embedding_cuda_source.attention_stem_source(
    temporal~,
    height~,
    width~,
    heads=32,
    head_dimension=64,
    rotary_dimension=48,
    register_rows=4,
    prefix~,
  )
  let rows = temporal * height * width + 5
  let hidden_bytes = rows.to_int64() * 2048L * 2L
  let qkv_bytes = hidden_bytes * 3L
  let qkv_offset = (hidden_bytes + 255L) / 256L * 256L
  let table_offset = (qkv_offset + qkv_bytes + 255L) / 256L * 256L
  let table_bytes = rows.to_int64() * 48L * 2L
  let scratch_bytes = table_offset + table_bytes
  guard scratch_bytes <= max_scratch_bytes else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  {
    source,
    symbols: ReadOnlyArray::from_array([
      prefix + "_norm",
      prefix + "_qkv",
      prefix + "_rotary_table",
      prefix + "_qk_rope",
    ]),
    rows,
    hidden_bytes,
    qkv_bytes,
    table_bytes,
    qkv_offset,
    table_offset,
    scratch_bytes,
    weights: @minimax_h3_weights.video_attention_weight_packing(
      layer~,
      max_bytes=50364416L,
    ),
  }
}

///|
/// Startup binding only. Scratch and output must be separate allocations from
/// input/weights and each other; caller retains all owners until queue close.
pub fn AttentionStem::launches(
  self : Self,
  functions : ArrayView[@device.Function],
  input : @device.Allocation,
  weights : @device.Allocation,
  scratch : @device.Allocation,
  output : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 4 && scratch.byte_count() >= self.scratch_bytes else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn region(
    buffer : @device.Allocation,
    offset : Int64,
    bytes : Int64,
    alignment : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      buffer,
      offset~,
      byte_count=bytes,
      alignment~,
    )
  }
  fn weight(i : Int) -> @device.KernelArgument raise @device.DeviceError {
    let copy = self.weights.copies[i]
    region(weights, copy.destination_offset, copy.byte_count, 4L)
  }
  fn launch(
    i : Int,
    grid : Int,
    block : Int,
    args : Array[@device.KernelArgument],
  ) -> @device.OrderedKernelLaunch raise @device.DeviceError {
    @device.OrderedKernelLaunch::new(
      functions[i],
      @device.KernelLaunchDimensions::new(
        grid_x=grid,
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
  let norm = region(scratch, 0L, self.hidden_bytes, 2L)
  let qkv = region(scratch, self.qkv_offset, self.qkv_bytes, 2L)
  let table = region(scratch, self.table_offset, self.table_bytes, 2L)
  [
    launch(0, self.rows, 1, [
      region(input, 0L, self.hidden_bytes * 2L, 4L),
      weight(0),
      norm,
    ]),
    launch(1, ((self.qkv_bytes / 2L + 255L) / 256L).to_int(), 256, [
      norm,
      weight(1),
      weight(2),
      weight(3),
      weight(4),
      weight(5),
      weight(6),
      qkv,
    ]),
    launch(2, ((self.table_bytes / 2L + 255L) / 256L).to_int(), 256, [table]),
    launch(3, self.rows * 32 * 3, 1, [
      qkv,
      table,
      region(output, 0L, self.qkv_bytes, 2L),
    ]),
  ]
}
