///|
/// Startup owner for an offline-compiled H3 block bundle. The caller owns the
/// admitted module, weights and external tensors. Keep this owner even when
/// prepare fails: close releases every successfully acquired resource.
pub struct Program {
  priv source : @minimax_h3_block_source.BlockSource
  priv functions : Array[@device.Function]
  priv mut workspace : @joint_diffusion_device.BlockWorkspace?
  priv mut ready : Bool
}

///|
pub fn Program::new(source : @minimax_h3_block_source.BlockSource) -> Self {
  { source, functions: [], workspace: None, ready: false, }
}

///|
/// The module must be the admitted AOT result of this exact source bundle.
/// No compilation, artifact hashing or file access occurs here.
pub fn Program::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
  max_workspace_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard !self.ready && self.functions.is_empty() && self.workspace is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let shape = self.source.shape
  let layout = @joint_diffusion_device.BlockWorkspaceLayout::new(
    rows=self.source.rows,
    hidden_width=shape.hidden_size(),
    head_count=shape.attention_head_count(),
    head_dimension=shape.attention_head_dimension(),
    intermediate_width=shape.feed_forward_dimension(),
    max_bytes=max_workspace_bytes,
  )
  acquire_functions(self.source.symbols[:], self.functions, name => {
    aot_module.load_function(@utf8.encode(name))
  })
  self.workspace = Some(
    @joint_diffusion_device.BlockWorkspace::allocate(context, layout),
  )
  self.ready = true
}

///|
/// Stop using prepared frames and release their leases before closing. A
/// failed close retains the failing handle for retry, rather than losing it.
pub fn Program::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  if self.workspace is Some(workspace) {
    workspace.close()
    self.workspace = None
  }
  release_functions(self.functions, function => function.close())
}

///|
fn[T] acquire_functions(
  symbols : ArrayView[String],
  handles : Array[T],
  load : (String) -> T raise @device.DeviceError,
) -> Unit raise @device.DeviceError {
  for symbol in symbols {
    handles.push(load(symbol))
  }
}

///|
fn[T] release_functions(
  handles : Array[T],
  close : (T) -> Unit raise @device.DeviceError,
) -> Unit raise @device.DeviceError {
  while !handles.is_empty() {
    close(handles[handles.length() - 1])
    let _ = handles.pop()
  }
}
