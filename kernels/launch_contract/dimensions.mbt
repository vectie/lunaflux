///|
const MAX_GRID_YZ : Int = 65535

///|
const MAX_BLOCK_DIMENSION : Int = 1024

///|
const MAX_THREADS_PER_BLOCK : Int = 1024

///|
const MAX_SHARED_MEMORY_BYTES : Int = 98304

///|
/// Exact bounded CUDA launch geometry retained as inert startup metadata.
pub struct AotLaunchDimensions {
  priv grid_x : Int
  priv grid_y : Int
  priv grid_z : Int
  priv block_x : Int
  priv block_y : Int
  priv block_z : Int
  priv shared_memory_bytes : Int
} derive(Debug, Eq)

///|
pub fn AotLaunchDimensions::new(
  grid_x~ : Int,
  grid_y~ : Int,
  grid_z~ : Int,
  block_x~ : Int,
  block_y~ : Int,
  block_z~ : Int,
  shared_memory_bytes~ : Int,
) -> AotLaunchDimensions raise LaunchContractError {
  if grid_x <= 0 ||
    grid_y <= 0 ||
    grid_z <= 0 ||
    grid_y > MAX_GRID_YZ ||
    grid_z > MAX_GRID_YZ {
    raise InvalidDimensions(InvalidGrid)
  }
  if block_x <= 0 ||
    block_y <= 0 ||
    block_z <= 0 ||
    block_x > MAX_BLOCK_DIMENSION ||
    block_y > MAX_BLOCK_DIMENSION ||
    block_z > MAX_BLOCK_DIMENSION {
    raise InvalidDimensions(InvalidBlock)
  }
  if block_x * block_y * block_z > MAX_THREADS_PER_BLOCK {
    raise InvalidDimensions(TooManyThreads)
  }
  if shared_memory_bytes < 0 || shared_memory_bytes > MAX_SHARED_MEMORY_BYTES {
    raise InvalidDimensions(InvalidSharedMemory)
  }
  { grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_memory_bytes }
}

///|
pub fn AotLaunchDimensions::grid_x(self : AotLaunchDimensions) -> Int {
  self.grid_x
}

///|
pub fn AotLaunchDimensions::grid_y(self : AotLaunchDimensions) -> Int {
  self.grid_y
}

///|
pub fn AotLaunchDimensions::grid_z(self : AotLaunchDimensions) -> Int {
  self.grid_z
}

///|
pub fn AotLaunchDimensions::block_x(self : AotLaunchDimensions) -> Int {
  self.block_x
}

///|
pub fn AotLaunchDimensions::block_y(self : AotLaunchDimensions) -> Int {
  self.block_y
}

///|
pub fn AotLaunchDimensions::block_z(self : AotLaunchDimensions) -> Int {
  self.block_z
}

///|
pub fn AotLaunchDimensions::shared_memory_bytes(
  self : AotLaunchDimensions,
) -> Int {
  self.shared_memory_bytes
}
