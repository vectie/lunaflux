///|
const MAX_KERNEL_ARGUMENTS : Int = 32

///|
const MAX_GRID_YZ : Int = 65535

///|
const MAX_BLOCK_DIMENSION : Int = 1024

///|
const MAX_THREADS_PER_BLOCK : Int = 1024

///|
const MAX_SHARED_MEMORY_BYTES : Int = 98304

///|
const MAX_KERNEL_ARGUMENT_ALIGNMENT : Int64 = 4096L

///|
fn invalid_launch_argument() -> DeviceError {
  InvalidArgument(AotKernelLaunch)
}

///|
fn is_valid_argument_alignment(alignment : Int64) -> Bool {
  alignment > 0L &&
  alignment <= MAX_KERNEL_ARGUMENT_ALIGNMENT &&
  (alignment & (alignment - 1L)) == 0L
}

///|
/// Bounded CUDA launch geometry. Hardware may impose a smaller dynamic shared
/// memory limit; that incompatibility remains a typed launch failure.
pub struct KernelLaunchDimensions {
  priv grid_x : Int
  priv grid_y : Int
  priv grid_z : Int
  priv block_x : Int
  priv block_y : Int
  priv block_z : Int
  priv shared_memory_bytes : Int
}

///|
pub fn KernelLaunchDimensions::new(
  grid_x~ : Int,
  grid_y~ : Int,
  grid_z~ : Int,
  block_x~ : Int,
  block_y~ : Int,
  block_z~ : Int,
  shared_memory_bytes~ : Int,
) -> KernelLaunchDimensions raise DeviceError {
  if grid_x <= 0 ||
    grid_y <= 0 ||
    grid_z <= 0 ||
    grid_y > MAX_GRID_YZ ||
    grid_z > MAX_GRID_YZ {
    raise invalid_launch_argument()
  }
  if block_x <= 0 ||
    block_y <= 0 ||
    block_z <= 0 ||
    block_x > MAX_BLOCK_DIMENSION ||
    block_y > MAX_BLOCK_DIMENSION ||
    block_z > MAX_BLOCK_DIMENSION {
    raise invalid_launch_argument()
  }
  let threads = block_x * block_y * block_z
  if threads > MAX_THREADS_PER_BLOCK ||
    shared_memory_bytes < 0 ||
    shared_memory_bytes > MAX_SHARED_MEMORY_BYTES {
    raise invalid_launch_argument()
  }
  { grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_memory_bytes }
}

///|
/// One checked device-memory region passed by address to an AOT kernel. The
/// region is revalidated by the native boundary immediately before launch.
pub struct KernelArgument {
  priv allocation : Allocation
  priv offset : Int64
  priv byte_count : Int64
  priv alignment : Int64
}

///|
pub fn KernelArgument::device_region(
  allocation : Allocation,
  offset~ : Int64,
  byte_count~ : Int64,
  alignment~ : Int64,
) -> KernelArgument raise DeviceError {
  if offset < 0L ||
    byte_count <= 0L ||
    !is_valid_argument_alignment(alignment) ||
    offset > allocation.byte_count ||
    byte_count > allocation.byte_count - offset {
    raise invalid_launch_argument()
  }
  { allocation, offset, byte_count, alignment }
}

///|
/// Immutable, reusable kernel parameter storage prepared outside the launch
/// path. Argument order is preserved exactly.
pub struct KernelArguments {
  priv allocations : FixedArray[@cuda.Allocation]
  priv offsets : FixedArray[Int64]
  priv byte_counts : FixedArray[Int64]
  priv alignments : FixedArray[Int64]
  priv length : Int
}

///|
pub fn KernelArguments::new(
  arguments : ArrayView[KernelArgument],
) -> KernelArguments raise DeviceError {
  if arguments.is_empty() || arguments.length() > MAX_KERNEL_ARGUMENTS {
    raise invalid_launch_argument()
  }
  let allocations : Array[@cuda.Allocation] = []
  let offsets : Array[Int64] = []
  let byte_counts : Array[Int64] = []
  let alignments : Array[Int64] = []
  for argument in arguments {
    allocations.push(argument.allocation.raw)
    offsets.push(argument.offset)
    byte_counts.push(argument.byte_count)
    alignments.push(argument.alignment)
  }
  {
    allocations: FixedArray::from_array(allocations),
    offsets: FixedArray::from_array(offsets),
    byte_counts: FixedArray::from_array(byte_counts),
    alignments: FixedArray::from_array(alignments),
    length: arguments.length(),
  }
}

///|
pub fn KernelArguments::length(self : KernelArguments) -> Int {
  self.length
}
