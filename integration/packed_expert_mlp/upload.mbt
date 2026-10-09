///|
/// One shared bounded upload effect for an immutable expert-bank plan.
/// Model adapters supply expert-major gate/up/down bindings. Weight payloads
/// never expand, and ownership of the bank remains with the caller.
pub fn upload_bank(
  context : @device.Context,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  plan : @precision.ExpertMlpPrecision,
  bank : @device.Allocation,
  weights~ : ArrayView[@materialize.PackedCheckpointWeight],
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard chunk_bytes > 0 && weights.length() == plan.local_experts().length() * 3 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let destinations = weights.mapi((i, weight) => {
    guard weight.representation() == plan.weights()[i % 3].representation() else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    (
      0,
      (i / 3).to_int64() * plan.expert_stride() + plan.weight_offsets()[i % 3],
    )
  })
  context.validate_allocation_region(
    bank,
    offset=0L,
    byte_count=plan.bank_bytes(),
    alignment=16L,
  )
  checkpoint.transfer_weights_to(
    root,
    weights,
    destinations~,
    destination_bytes=[plan.bank_bytes()],
    scratch=FixedArray::make(chunk_bytes, b'\x00'),
    limits~,
    consume=(_, offset, buffer, count) => {
      try
        context.copy_from_fixed_host(bank, buffer, 0L, offset, count.to_int64())
      catch {
        _ => false
      } noraise {
        _ => true
      }
    },
  )
}
