///|
/// Startup-only translation failures. Successful upload remains component
/// scoped and cannot represent a complete MiniMax H3 model without both VAEs.
pub(all) suberror MiniMaxH3DeviceUploadError {
  Host(@minimax_h3_host_materialization.MiniMaxH3HostMaterializationError)
  Borrow(@segmented_host_borrow.SegmentedHostBorrowError)
  Device(@segmented_device_materialize.SegmentedDeviceMaterializationError)
} derive(Debug, Eq)

///|
fn[T] from_host(
  body : () -> T raise @minimax_h3_host_materialization.MiniMaxH3HostMaterializationError,
) -> T raise MiniMaxH3DeviceUploadError {
  body() catch {
    error => raise Host(error)
  }
}

///|
fn[T] from_borrow(
  body : () -> T raise @segmented_host_borrow.SegmentedHostBorrowError,
) -> T raise MiniMaxH3DeviceUploadError {
  body() catch {
    error => raise Borrow(error)
  }
}

///|
fn[T] from_device(
  body : () -> T raise @segmented_device_materialize.SegmentedDeviceMaterializationError,
) -> T raise MiniMaxH3DeviceUploadError {
  body() catch {
    error => raise Device(error)
  }
}
