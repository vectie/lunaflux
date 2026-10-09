///|
fn dimensions(
  x : Int,
  y : Int,
  threads : Int,
) -> @device.KernelLaunchDimensions raise @device.DeviceError {
  @device.KernelLaunchDimensions::new(
    grid_x=x,
    grid_y=y,
    grid_z=1,
    block_x=threads,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn region(
  a : @device.Allocation,
  offset : Int64,
  bytes : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    a,
    offset~,
    byte_count=bytes,
    alignment=2L,
  )
}

///|
/// Pure layer recipe shared by binding and sequence tests.
fn layer_function_order() -> ReadOnlyArray[Int] {
  [1, 2, 3, 3, 4, 5, 6, 7, 10, 1, 8, 9, 10]
}

///|
/// Bind the full text-only feature extractor, without an LM head, final norm,
/// sampling, or a decode KV cache. All allocations/functions are borrowed and
/// must outlive the consuming ordered executor. Counts[3] must equal plan.rows;
/// token IDs must pass plan.validate_text_tokens before their upload. Cos/sin
/// are BF16 tables prepared from the admitted text encoder's rotary config.
pub fn TextEncoderSource::bind(
  self : Self,
  functions : ArrayView[@device.Function],
  layers : ArrayView[(@text_plan.LayerWeights, @device.Allocation)],
  counts~ : @device.Allocation,
  token_ids~ : @device.Allocation,
  embedding~ : @device.Allocation,
  workspace~ : @device.Allocation,
  cosine~ : @device.Allocation,
  sine~ : @device.Allocation,
  visual? : (Int, @device.Allocation, @device.Allocation),
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard functions.length() == 13 && layers.length() == 50 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  for i, (layout, _) in layers {
    guard layout.layer == i else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  let p = self.plan
  let regions = p.workspace_regions()
  let args = regions.map(r => region(workspace, r.0, r.1))
  let state = args[0]
  let norm = args[1]
  let q = args[2]
  let k = args[3]
  let v = args[4]
  let attention = args[5]
  let branch = args[6]
  let product = args[7]
  let count = @device.KernelArgument::device_region(
    counts,
    offset=0L,
    byte_count=16L,
    alignment=4L,
  )
  let rope_bytes = p.rows.to_int64() * 128L * 2L
  let cos = region(cosine, 0L, rope_bytes)
  let sin = region(sine, 0L, rope_bytes)
  let launches : Array[@device.OrderedKernelLaunch] = []
  fn append(
    index : Int,
    grid : @device.KernelLaunchDimensions,
    operands : Array[@device.KernelArgument],
  ) -> Unit raise @device.DeviceError {
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[index],
        grid,
        @device.KernelArguments::new(operands),
      ),
    )
  }
  fn linear(
    width : Int,
  ) -> @device.KernelLaunchDimensions raise @device.DeviceError {
    dimensions(
      ((p.rows.to_int64() * width.to_int64() + 255L) / 256L).to_int(),
      1,
      256,
    )
  }
  append(0, linear(5120), [
    count,
    @device.KernelArgument::device_region(
      token_ids,
      offset=0L,
      byte_count=p.rows.to_int64() * 4L,
      alignment=4L,
    ),
    region(embedding, 0L, 151936L * 5120L * 2L),
    state,
  ])
  fn visual_stage(index : Int) -> Unit raise @device.DeviceError {
    if visual is Some((rows, positions, features)) {
      guard rows > 0 && rows <= p.rows else {
        raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
      }
      let bytes = rows.to_int64() * 5120L * 2L
      append(if index == 0 { 11 } else { 12 }, linear(5120), [
        count,
        @device.KernelArgument::device_region(
          positions,
          offset=0L,
          byte_count=rows.to_int64() * 4L,
          alignment=4L,
        ),
        region(features, index.to_int64() * bytes, bytes),
        state,
      ])
    }
  }
  visual_stage(0)
  for (layout, weights) in layers {
    let w = layout.regions.map(r => region(weights, r.offset, r.bytes))
    let up_gate = region(
      weights,
      layout.regions[8].offset,
      layout.regions[8].bytes + layout.regions[9].bytes,
    )
    let steps : Array[
      (@device.KernelLaunchDimensions, Array[@device.KernelArgument]),
    ] = [
      (dimensions(p.rows, 1, 1), [state, w[0], norm]),
      (linear(8192), [count, norm, w[1], q]),
      (linear(1024), [count, norm, w[2], k]),
      (linear(1024), [count, norm, w[3], v]),
      (dimensions(p.rows, 72, 1), [q, k, w[4], w[5]]),
      (dimensions(p.rows, 72, 64), [q, k, cos, sin]),
      (dimensions(p.rows, 64, 1), [q, k, v, attention]),
      (linear(5120), [count, attention, w[6], branch]),
      (linear(5120), [state, branch, state]),
      (dimensions(p.rows, 1, 1), [state, w[7], norm]),
      (linear(25600), [norm, up_gate, product]),
      (linear(5120), [product, w[10], branch]),
      (linear(5120), [state, branch, state]),
    ]
    for i, function in layer_function_order() {
      let (grid, operands) = steps[i]
      append(function, grid, operands)
    }
    if deepstack_stage(layout.layer) is Some(index) {
      visual_stage(index)
    }
  }
  launches
}
