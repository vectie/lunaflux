///|
pub(all) struct Modules {
  common : @device.Module
  prefill : @device.Module
  decode : @device.Module
}

///|
pub(all) struct Metadata {
  counts : @device.Allocation
  canvas_positions : @device.Allocation
  causal_positions : @device.Allocation
  row_offsets : @device.Allocation
  sequence_lengths : @device.Allocation
  page_offsets : @device.Allocation
  physical_pages : @device.Allocation
}

///|
pub(all) struct Weights {
  query : @device.Allocation
  key : @device.Allocation
  value : @device.Allocation
  output : @device.Allocation
  query_norm : @device.Allocation
  key_norm : @device.Allocation
}

///|
/// Borrowed persistent cache planes. Offsets allow views into the existing
/// multi-layer KV arena; separate plane allocations use the default offsets.
pub struct Caches {
  priv key : @device.Allocation
  priv value : @device.Allocation
  priv key_offset : Int64
  priv value_offset : Int64
}

///|
pub fn Caches::new(
  key : @device.Allocation,
  value : @device.Allocation,
  key_offset~ : Int64 = 0L,
  value_offset~ : Int64 = 0L,
) -> Self {
  { key, value, key_offset, value_offset, }
}

///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns function handles and prepared launches for both phases. Modules,
/// metadata, weights, input/output, workspace and persistent caches are borrowed.
/// Close enclosing queues before this frame, then release borrowed resources.
pub struct Frame {
  priv program : Program
  priv functions : Array[@device.Function]
  priv prefill : Array[@device.OrderedKernelLaunch]
  priv decode : Array[@device.OrderedKernelLaunch]
  priv mut status : FrameState
}

///|
pub fn Frame::new(program : Program) -> Self {
  { program, functions: [], prefill: [], decode: [], status: Fresh, }
}

///|
/// Prepare once. The caller supplies valid StepCounts/CSR/page ownership and
/// separate THW rotary coordinates. Cache writes precede reads in queue order.
/// Close a partially prepared frame on failure before releasing its resources.
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  modules~ : Modules,
  metadata~ : Metadata,
  weights~ : Weights,
  caches~ : Caches,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  workspace~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  context.validate_allocation_region(
    workspace,
    offset=0L,
    byte_count=self.program.workspace_bytes(),
    alignment=16L,
  )
  let bytes = self.program.region(KeyCache).byte_count
  context.validate_allocation_region(
    caches.key,
    offset=caches.key_offset,
    byte_count=bytes,
    alignment=2L,
  )
  context.validate_allocation_region(
    caches.value,
    offset=caches.value_offset,
    byte_count=bytes,
    alignment=2L,
  )
  guard !caches.key.same_storage(caches.value) ||
    caches.key_offset >= caches.value_offset + bytes ||
    caches.value_offset >= caches.key_offset + bytes else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  let borrowed = [
    metadata.counts,
    metadata.canvas_positions,
    metadata.causal_positions,
    metadata.row_offsets,
    metadata.sequence_lengths,
    metadata.page_offsets,
    metadata.physical_pages,
    weights.query,
    weights.key,
    weights.value,
    weights.output,
    weights.query_norm,
    weights.key_norm,
    caches.key,
    caches.value,
    input,
    output,
  ]
  guard borrowed.all(buffer => !buffer.same_storage(workspace)) else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let buffer = fn(slot) {
    match slot {
      Counts => metadata.counts
      CanvasPositions => metadata.canvas_positions
      CausalPositions => metadata.causal_positions
      RowOffsets => metadata.row_offsets
      SequenceLengths => metadata.sequence_lengths
      PageOffsets => metadata.page_offsets
      PhysicalPages => metadata.physical_pages
      Input => input
      Output => output
      QueryWeight => weights.query
      KeyWeight => weights.key
      ValueWeight => weights.value
      OutputWeight => weights.output
      QueryNormWeight => weights.query_norm
      KeyNormWeight => weights.key_norm
      KeyCache => caches.key
      ValueCache => caches.value
      _ => workspace
    }
  }
  let loaded : Map[String, @device.Function] = Map([])
  for phase in [Phase::Prefill, Decode] {
    let queue = match phase {
      Prefill => self.prefill
      Decode => self.decode
    }
    for stage in self.program.stages(phase) {
      let function = match loaded.get(stage.symbol()) {
        Some(function) => function
        None => {
          let aot = if stage.is_attention() {
            match phase {
              Prefill => modules.prefill
              Decode => modules.decode
            }
          } else {
            modules.common
          }
          let function = aot.load_function(@utf8.encode(stage.symbol()))
          self.functions.push(function)
          loaded[stage.symbol()] = function
          function
        }
      }
      let arguments = stage
        .arguments()
        .map(slot => {
          let region = self.program.region(slot)
          let offset = match slot {
            KeyCache => caches.key_offset
            ValueCache => caches.value_offset
            _ => region.workspace_offset.unwrap_or(0L)
          }
          @device.KernelArgument::device_region(
            buffer(slot),
            offset~,
            byte_count=region.byte_count,
            alignment=region.alignment,
          )
        })
      let (grid_x, grid_y, block_x, shared_memory_bytes) = stage.launch()
      queue.push(
        @device.OrderedKernelLaunch::new(
          function,
          @device.KernelLaunchDimensions::new(
            grid_x~,
            grid_y~,
            grid_z=1,
            block_x~,
            block_y=1,
            block_z=1,
            shared_memory_bytes=shared_memory_bytes.to_int(),
          ),
          @device.KernelArguments::new(arguments),
        ),
      )
    }
  }
  self.status = Prepared
}

///|
pub fn Frame::launches(
  self : Self,
  phase : Phase,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  match phase {
    Prefill => self.prefill[:]
    Decode => self.decode[:]
  }
}

///|
pub fn Frame::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.prefill.clear()
  self.decode.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  self.status = Closed
}
