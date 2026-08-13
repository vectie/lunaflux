///|
/// Bounded CUDA loader state. Numeric driver codes and vendor messages never
/// enter this representation.
pub(all) enum DriverAvailability {
  Available
  UnsupportedPlatform
  DriverLibraryMissing
  DriverAbiIncomplete
  DriverInitializationFailed
  NoDevices
  InventoryFailed
} derive(Debug, Eq)

///|
pub struct DriverInventory {
  availability : DriverAvailability
  driver_version : Int
  device_count : Int
  cublas_lt_available : Bool
} derive(Debug, Eq)

///|
pub fn DriverInventory::availability(
  self : DriverInventory,
) -> DriverAvailability {
  self.availability
}

///|
pub fn DriverInventory::driver_version(self : DriverInventory) -> Int {
  self.driver_version
}

///|
pub fn DriverInventory::device_count(self : DriverInventory) -> Int {
  self.device_count
}

///|
pub fn DriverInventory::cublas_lt_available(self : DriverInventory) -> Bool {
  self.cublas_lt_available
}

///|
pub struct RawDeviceInfo {
  total_memory_bytes : Int64
  compute_major : Int
  compute_minor : Int
  supports_bf16 : Bool
  cublas_lt_available : Bool
  name : String
} derive(Debug, Eq)

///|
pub fn RawDeviceInfo::total_memory_bytes(self : RawDeviceInfo) -> Int64 {
  self.total_memory_bytes
}

///|
pub fn RawDeviceInfo::compute_major(self : RawDeviceInfo) -> Int {
  self.compute_major
}

///|
pub fn RawDeviceInfo::compute_minor(self : RawDeviceInfo) -> Int {
  self.compute_minor
}

///|
pub fn RawDeviceInfo::supports_bf16(self : RawDeviceInfo) -> Bool {
  self.supports_bf16
}

///|
pub fn RawDeviceInfo::cublas_lt_available(self : RawDeviceInfo) -> Bool {
  self.cublas_lt_available
}

///|
pub fn RawDeviceInfo::name(self : RawDeviceInfo) -> String {
  self.name
}

///|
pub(all) enum CudaOperation {
  DeviceInventory
  ContextCreate
  ContextClose
  StreamCreate
  StreamClose
  StreamSynchronize
  EventCreate
  EventClose
  EventRecord
  EventSynchronize
  EventElapsed
  AllocationCreate
  AllocationClose
  AllocationLeaseCreate
  AllocationLeaseClose
  AllocationRegionValidate
  CopyToDevice
  CopyToHost
  ModuleLoad
  ModuleClose
  FunctionLoad
  FunctionClose
  AotKernelLaunch
  CublasLtCreate
  CublasLtClose
  Bf16GemmPlanCreate
  Bf16GemmPlanClose
  Bf16GemmRun
} derive(Debug, Eq)

///|
/// Payload-safe internal failure. It deliberately omits CUDA result numbers,
/// native pointers, library paths, and vendor-provided strings.
pub(all) suberror CudaError {
  Unavailable(CudaOperation)
  InvalidArgument(CudaOperation)
  Closed(CudaOperation)
  Busy(CudaOperation)
  DriverFailure(CudaOperation)
  HostAllocationFailed(CudaOperation)
  SizeOverflow(CudaOperation)
  Unsupported(CudaOperation)
  InvalidOutput(CudaOperation)
} derive(Debug, Eq)
