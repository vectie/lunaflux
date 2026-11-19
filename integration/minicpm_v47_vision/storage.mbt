///|
pub(all) struct BufferRegion {
  allocation : @device.Allocation
  offset : Int64
  byte_count : Int64
  alignment : Int64
}

///|
/// Borrowed views into one Storage owner. Scratch views intentionally share an
/// allocation; all pixels/metadata/features use distinct allocations.
pub struct Buffers {
  priv regions : ReadOnlyArray[(Plane, BufferRegion)]
  priv allocations : ReadOnlyArray[@device.Allocation]
  priv rows : (Int32, Int32, Int32)
}

///|
pub fn Buffers::region(
  self : Self,
  plane : Plane,
) -> BufferRegion raise @device.DeviceError {
  for i in 0..<self.regions.length() {
    let (key, region) = self.regions[i]
    if key == plane {
      return region
    }
  }
  raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
}

///|
fn Program::external_regions(self : Self) -> Array[(Plane, Int64, Int64)] {
  let (raw, encoded, output) = self.rows
  [
    (Pixels, raw.to_int64() * 588L * 2L, 2L),
    (Features, output.to_int64() * 2048L * 2L, 2L),
    (RawCounts, 16L, 4L),
    (EncodedCounts, 16L, 4L),
    (OutputCounts, 16L, 4L),
    (PositionIds, raw.to_int64() * 4L, 4L),
    (RawStarts, raw.to_int64() * 4L, 4L),
    (RawEnds, raw.to_int64() * 4L, 4L),
    (EncodedStarts, encoded.to_int64() * 4L, 4L),
    (EncodedEnds, encoded.to_int64() * 4L, 4L),
    (WindowStarts, raw.to_int64() * 4L, 4L),
    (WindowEnds, raw.to_int64() * 4L, 4L),
    (WindowIndices, raw.to_int64() * 4L, 4L),
    (MergerIndices, encoded.to_int64() * 4L, 4L),
  ]
}

///|
/// Complete mutable allocation payload; excludes weights, modules and queue
/// overhead. There are15 allocations, including the shared scratch arena.
pub fn Program::storage_bytes(self : Self) -> Int64 {
  self
  .external_regions()
  .fold(init=self.workspace_bytes, (total, entry) => total + entry.1)
}

///|
priv enum StorageState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
pub struct Storage {
  priv program : Program
  priv allocations : Array[@device.Allocation]
  priv mut bindings : Buffers?
  priv mut state : StorageState
}

///|
pub fn Storage::new(program : Program) -> Self {
  { program, allocations: [], bindings: None, state: Fresh, }
}

///|
pub fn Storage::prepare(
  self : Self,
  context : @device.Context,
  maximum_device_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.state is Fresh &&
    maximum_device_bytes >= self.program.storage_bytes() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.state = Failed
  let regions : Array[(Plane, BufferRegion)] = []
  let arena = context.allocate(self.program.workspace_bytes())
  self.allocations.push(arena)
  for region in self.program.regions {
    regions.push(
      (
        region.plane,
        {
          allocation: arena,
          offset: region.offset,
          byte_count: region.bytes,
          alignment: 16L,
        },
      ),
    )
  }
  for (plane, bytes, alignment) in self.program.external_regions() {
    let allocation = context.allocate(bytes)
    self.allocations.push(allocation)
    regions.push(
      (plane, { allocation, offset: 0L, byte_count: bytes, alignment, }),
    )
  }
  self.bindings = Some({
    regions: ReadOnlyArray::from_array(regions),
    allocations: ReadOnlyArray::from_array(self.allocations.copy()),
    rows: self.program.rows,
  })
  // Zero live counts before optional graph preparation. Capacity padding never
  // needs initialization because every emitted stage checks its live count.
  let zero = FixedArray::make(16, b'\x00')
  for plane in [RawCounts, EncodedCounts, OutputCounts] {
    let region = self.bindings.unwrap().region(plane)
    context.copy_from_fixed_host(region.allocation, zero, 0L, 0L, 16L)
  }
  self.state = Prepared
}

///|
pub fn Storage::buffers(self : Self) -> Buffers raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.bindings.unwrap()
}

///|
pub fn Storage::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.bindings = None
  while self.allocations.last() is Some(allocation) {
    allocation.close()
    ignore(self.allocations.pop())
  }
  self.state = Closed
}
