///|
/// Startup-only translation failures. Successful upload yields raw device
/// weights, never kernel qualification or execution authority.
pub(all) suberror Glm53DeviceUploadError {
  Host(@glm53_host_materialize.Glm53HostMaterializeError)
  Borrow(@segmented_host_borrow.SegmentedHostBorrowError)
  Device(@segmented_device_materialize.SegmentedDeviceMaterializationError)
} derive(Debug, Eq)

///|
fn[T] from_host(
  body : () -> T raise @glm53_host_materialize.Glm53HostMaterializeError,
) -> T raise Glm53DeviceUploadError {
  body() catch {
    error => raise Host(error)
  }
}

///|
fn[T] from_borrow(
  body : () -> T raise @segmented_host_borrow.SegmentedHostBorrowError,
) -> T raise Glm53DeviceUploadError {
  body() catch {
    error => raise Borrow(error)
  }
}

///|
fn[T] from_device(
  body : () -> T raise @segmented_device_materialize.SegmentedDeviceMaterializationError,
) -> T raise Glm53DeviceUploadError {
  body() catch {
    error => raise Device(error)
  }
}
