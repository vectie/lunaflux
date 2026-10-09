///|
/// Startup-only translation failures. Successful upload yields raw device
/// weights, never kernel qualification or execution authority.
pub(all) suberror DeepSeekV4DeviceUploadError {
  Host(@deepseek_v4_host_materialization.DeepSeekV4HostMaterializationError)
  TokenHash(
    @deepseek_v4_host_materialization.DeepSeekV4TokenHashMaterializationError
  )
  Borrow(@segmented_host_borrow.SegmentedHostBorrowError)
  Device(@segmented_device_materialize.SegmentedDeviceMaterializationError)
  TokenHashSourceReleased
  TokenHashLayerCountMismatch(Int)
  TokenHashLayerOrderMismatch(ordinal~ : Int, layer~ : Int)
  TokenHashTableByteCountMismatch(ordinal~ : Int, actual~ : Int)
  TokenHashMetadataDigestInvalid(Int)
  TokenHashIdentityMismatch
  TokenHashSourceDigestMismatch
  TokenHashLayoutMismatch
} derive(Debug, Eq)

///|
fn[T] from_host(
  body : () -> T raise @deepseek_v4_host_materialization.DeepSeekV4HostMaterializationError,
) -> T raise DeepSeekV4DeviceUploadError {
  body() catch {
    error => raise Host(error)
  }
}

///|
fn[T] from_borrow(
  body : () -> T raise @segmented_host_borrow.SegmentedHostBorrowError,
) -> T raise DeepSeekV4DeviceUploadError {
  body() catch {
    error => raise Borrow(error)
  }
}

///|
fn[T] from_token_hash(
  body : () -> T raise @deepseek_v4_host_materialization.DeepSeekV4TokenHashMaterializationError,
) -> T raise DeepSeekV4DeviceUploadError {
  body() catch {
    error => raise TokenHash(error)
  }
}

///|
fn[T] from_device(
  body : () -> T raise @segmented_device_materialize.SegmentedDeviceMaterializationError,
) -> T raise DeepSeekV4DeviceUploadError {
  body() catch {
    error => raise Device(error)
  }
}
