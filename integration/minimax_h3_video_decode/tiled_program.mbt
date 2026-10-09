///|
/// Sequential tiles reuse one decoder, one gathered latent tile and one set of
/// raw spatial outputs; only two full 28-frame clips survive for temporal blend.
pub struct TiledDecoderProgram {
  priv plan : TiledDecoder
  priv tile : TileDecoderProgram
  priv functions : Array[@device.Function]
  priv buffers : Array[@device.Allocation]
  priv mut ready : Bool
}

///|
pub fn TiledDecoderProgram::new(plan : TiledDecoder) -> Self {
  {
    plan,
    tile: TileDecoderProgram::new(plan.tile),
    functions: [],
    buffers: [],
    ready: false,
  }
}

///|
pub fn TiledDecoderProgram::plan(self : Self) -> TiledDecoder {
  self.plan
}

///|
pub fn TiledDecoderProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
) -> Unit raise @device.DeviceError {
  guard !self.ready && self.functions.is_empty() && self.buffers.is_empty() else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  self.tile.prepare(context, aot_module)
  for name in self.plan.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(name)))
  }
  for
    bytes in [
      self.plan.gather_bytes,
      self.plan.tile_store_bytes,
      self.plan.clip_bytes,
      self.plan.clip_bytes,
      self.plan.output_bytes,
    ] {
    self.buffers.push(context.allocate(bytes))
  }
  self.ready = true
}

///|
/// Bind the full ordered request once. No output is exposed as completed here;
/// the request owner must await queue completion and close it before release.
pub fn TiledDecoderProgram::bind(
  self : Self,
  input : @device.Allocation,
  input_offset? : Int64 = 0L,
  frontend_weights : @device.Allocation,
  blocks : ArrayView[@device.Allocation],
  head_weights : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.ready && self.buffers.length() == 5 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  fn region(
    a : @device.Allocation,
    o : Int64,
    n : Int64,
  ) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      a,
      offset=o,
      byte_count=n,
      alignment=4L,
    )
  }
  fn launch(
    i : Int,
    bytes : Int64,
    args : Array[@device.KernelArgument],
  ) -> @device.OrderedKernelLaunch raise @device.DeviceError {
    @device.OrderedKernelLaunch::new(
      self.functions[i],
      dimensions(bytes),
      @device.KernelArguments::new(args),
    )
  }
  let launches : Array[@device.OrderedKernelLaunch] = []
  let gather = self.buffers[0]
  let store = self.buffers[1]
  for chunk in 0..<self.plan.temporal.chunks {
    for index in 0..<self.plan.tile_count {
      launches.push(
        launch(
          2 + chunk * self.plan.tile_count + index,
          self.plan.gather_bytes,
          [
            region(input, input_offset, self.plan.input_bytes),
            region(gather, 0L, self.plan.gather_bytes),
          ],
        ),
      )
      for
        step in self.tile.bind(
          gather,
          input_offset=0L,
          frontend_weights,
          blocks,
          head_weights,
        ) {
        launches.push(step)
      }
      self.tile.with_pixels(pixels => {
        launches.push(
          launch(0, self.plan.tile.head.output_bytes, [
            region(pixels, 0L, self.plan.tile.head.output_bytes),
            region(
              store,
              index.to_int64() * self.plan.tile.head.output_bytes,
              self.plan.tile.head.output_bytes,
            ),
          ]),
        )
      })
    }
    let current = self.buffers[2 + chunk % 2]
    let previous = self.buffers[2 + (chunk + 1) % 2]
    launches.push(
      launch(1, self.plan.clip_bytes, [
        region(store, 0L, self.plan.tile_store_bytes),
        region(current, 0L, self.plan.clip_bytes),
      ]),
    )
    let parts = if chunk == self.plan.temporal.chunks - 1 { 22L } else { 17L }
    launches.push(
      launch(
        2 + self.plan.temporal.chunks * self.plan.tile_count + chunk,
        self.plan.clip_bytes / 28L * parts,
        [
          region(current, 0L, self.plan.clip_bytes),
          region(previous, 0L, self.plan.clip_bytes),
          region(self.buffers[4], 0L, self.plan.output_bytes),
        ],
      ),
    )
  }
  launches
}

///|
/// Normalized F32 [3,output_frames,height,width], before ImageNet de-normalize.
pub fn[T] TiledDecoderProgram::with_pixels(
  self : Self,
  action : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.ready && self.buffers.length() == 5 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  action(self.buffers[4])
}

///|
pub fn TiledDecoderProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.ready = false
  self.tile.close()
  while !self.buffers.is_empty() {
    self.buffers[self.buffers.length() - 1].close()
    let _ = self.buffers.pop()
  }
  while !self.functions.is_empty() {
    self.functions[self.functions.length() - 1].close()
    let _ = self.functions.pop()
  }
}
