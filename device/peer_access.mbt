///|
/// Reports directed physical peer-access capability between two visible CUDA
/// ordinals. This inventory query creates no context and grants no execution
/// or peer-enable authority.
pub fn can_access_peer(
  source_ordinal : Int,
  destination_ordinal : Int,
) -> Bool raise DeviceError {
  @cuda.device_can_access_peer(source_ordinal, destination_ordinal) catch {
    error => raise error_from_cuda_for(error, DevicePeerAccess)
  }
}
