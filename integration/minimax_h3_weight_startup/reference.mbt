///|
/// Existing reference encoder bindings address the complete segmented VAE
/// tensor table. Reuse its planner and uploader, not a second packing ABI.
/// Append this descriptor to the decoder's component list to upload both in
/// one host materialization. The 64-byte padding bound is conservative.
pub fn reference_encoder_packing(
  manifest : @host.MiniMaxH3HostComponentManifest,
  max_bytes~ : Int64,
) -> Packing raise {
  guard manifest.component() == AudioVae || manifest.component() == VideoVae else {
    raise InvalidPacking
  }
  let bytes = reference_reservation(
    manifest.total_tensor_bytes(),
    manifest.tensor_count(),
    max_bytes,
  )
  let mut maximum = 0L
  let mut total = 0L
  for i in 0..<manifest.arena_count() {
    guard manifest.arena_size(i) is Some(n) else { raise InvalidPacking }
    guard n > 0L && n <= 2147483647L && n <= 9223372036854775807L - total else {
      raise BudgetExceeded
    }
    maximum = Int64::max(maximum, n)
    total += n
  }
  let limits = @segmented.SegmentedDeviceMaterializationLimits::new(
    alignment_bytes=64L,
    maximum_source_arenas=manifest.arena_count(),
    maximum_regions=manifest.tensor_count(),
    maximum_host_arena_bytes=maximum.to_int(),
    maximum_total_source_bytes=total,
    maximum_device_arena_bytes=bytes,
  )
  ReferenceEncoder(limits, bytes)
}

///|
fn reference_reservation(
  payload : Int64,
  count : Int,
  maximum : Int64,
) -> Int64 raise StartupError {
  guard payload > 0L && count > 0 && payload <= maximum else {
    raise BudgetExceeded
  }
  let padding = count.to_int64() * 63L
  guard padding <= maximum - payload else { raise BudgetExceeded }
  payload + padding
}
