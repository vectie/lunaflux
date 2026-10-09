///|
/// Complete single-tile decoder. Spatial/temporal stitching is an outer plan.
pub struct TileDecoder {
  source : Bytes
  frontend : Frontend
  block : Block
  head : Head
  block_weights : ReadOnlyArray[@minimax_h3_weights.VideoBlockWeightPacking]
  workspace_bytes : Int64
  scratch_bytes : Int64
}

///|
pub fn TileDecoder::new(
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  prefix~ : String,
  max_workspace_bytes~ : Int64,
) -> Self raise {
  let frontend = Frontend::new(
    batch=1,
    temporal~,
    height~,
    width~,
    prefix=prefix + "_front",
    max_workspace_bytes~,
  )
  let block = Block::new(
    temporal~,
    height~,
    width~,
    prefix=prefix + "_block",
    max_scratch_bytes=max_workspace_bytes,
  )
  let head = Head::new(temporal~, height~, width~, prefix=prefix + "_last")
  let scratch_bytes = if block.scratch_bytes > head.hidden_bytes {
    block.scratch_bytes
  } else {
    head.hidden_bytes
  }
  let workspace_bytes = frontend.input_bytes +
    frontend.output_bytes * 2L +
    scratch_bytes +
    head.output_bytes
  guard workspace_bytes <= max_workspace_bytes else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let block_weights = Array::makei(36, layer => {
    @minimax_h3_weights.video_block_weight_packing(layer~, max_bytes=300000000L)
  })
  {
    source: frontend.source + block.source + head.source,
    frontend,
    block,
    head,
    block_weights: ReadOnlyArray::from_array(block_weights),
    workspace_bytes,
    scratch_bytes,
  }
}

///|
/// Owns all working tensors, admitted functions and the frontend. Weights and
/// source module are borrowed. Keep owner after any failed prepare or close.
pub struct TileDecoderProgram {
  priv plan : TileDecoder
  priv frontend : FrontendProgram
  priv functions : Array[@device.Function]
  priv mut ping : @device.Allocation?
  priv mut scratch : @device.Allocation?
  priv mut pixels : @device.Allocation?
  priv mut ready : Bool
}

///|
pub fn TileDecoderProgram::new(plan : TileDecoder) -> Self {
  {
    plan,
    frontend: FrontendProgram::new(plan.frontend),
    functions: [],
    ping: None,
    scratch: None,
    pixels: None,
    ready: false,
  }
}

///|
pub fn TileDecoderProgram::plan(self : Self) -> TileDecoder {
  self.plan
}

///|
pub fn TileDecoderProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
) -> Unit raise @device.DeviceError {
  guard !self.ready &&
    self.functions.is_empty() &&
    self.ping is None &&
    self.scratch is None &&
    self.pixels is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  self.frontend.prepare(context, aot_module)
  for symbol in self.plan.block.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(symbol)))
  }
  for symbol in self.plan.head.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(symbol)))
  }
  self.ping = Some(context.allocate(self.plan.frontend.output_bytes))
  self.scratch = Some(context.allocate(self.plan.scratch_bytes))
  self.pixels = Some(context.allocate(self.plan.head.output_bytes))
  self.ready = true
}

///|
/// Exactly 400 launches: frontend 2, 36 blocks × 11, head/unpack 2.
/// Caller submits this list once through a prepared ordered queue.
pub fn TileDecoderProgram::bind(
  self : Self,
  input : @device.Allocation,
  input_offset~ : Int64,
  frontend_weights : @device.Allocation,
  blocks : ArrayView[@device.Allocation],
  head_weights : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.ready &&
    self.ping is Some(ping) &&
    self.scratch is Some(scratch) &&
    self.pixels is Some(pixels) &&
    blocks.length() == 36 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let launches = self.frontend.bind(input, input_offset~, frontend_weights)
  self.frontend.with_output(hidden => {
    for layer in 0..<36 {
      let (source, destination) = if layer % 2 == 0 {
        (hidden, ping)
      } else {
        (ping, hidden)
      }
      let block_launches = self.plan.block.launches(
        self.functions[:9],
        self.plan.block_weights[layer],
        source,
        blocks[layer],
        scratch,
        destination,
      )
      for launch in block_launches {
        launches.push(launch)
      }
    }
    for
      launch in self.plan.head.launches(
        self.functions[9:],
        hidden,
        head_weights,
        scratch,
        pixels,
      ) {
      launches.push(launch)
    }
  })
  launches
}

///|
/// Borrow for downstream queue binding/readback after caller-observed completion.
pub fn[T] TileDecoderProgram::with_pixels(
  self : Self,
  action : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.ready && self.pixels is Some(pixels) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  action(pixels)
}

///|
pub fn TileDecoderProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  if self.pixels is Some(a) {
    a.close()
    self.pixels = None
  }
  if self.scratch is Some(a) {
    a.close()
    self.scratch = None
  }
  if self.ping is Some(a) {
    a.close()
    self.ping = None
  }
  self.frontend.close()
  while !self.functions.is_empty() {
    self.functions[self.functions.length() - 1].close()
    let _ = self.functions.pop()
  }
}
