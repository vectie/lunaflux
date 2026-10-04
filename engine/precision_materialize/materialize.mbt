///|
pub(all) suberror MaterializeError {
  WrongPlan
  Conversion(@format.PrecisionError)
} derive(Debug, Eq)

///|
/// Startup binding only. The model loader must authenticate source bytes before
/// constructing PackedTensor. Serving retains only the resulting BF16 arena;
/// it does not repeat validation or call a scalar codec on token steps.
pub struct StartupWeight {
  priv source : @format.PackedTensor
  priv bytes : Int64
}

///|
pub fn StartupWeight::new(
  plan : @precision.OperandPlan,
  source : @format.PackedTensor,
) -> StartupWeight raise MaterializeError {
  if plan.role() != Weight ||
    plan.placement() != StartupExpandBf16 ||
    plan.effect() != StartupWrite ||
    plan.representation() != source.representation() {
    raise WrongPlan
  }
  let (bytes, scratch) = plan.memory()
  if scratch != 0L {
    raise WrongPlan
  }
  { source, bytes, }
}

///|
pub fn StartupWeight::bytes(self : StartupWeight) -> Int64 {
  self.bytes
}

///|
/// Caller chooses a bounded upload chunk and owns its lifetime. This path
/// deliberately uses more resident memory than native compact-weight kernels.
pub fn StartupWeight::write_chunk(
  self : StartupWeight,
  first_element : Int64,
  count : Int,
  destination : FixedArray[Byte],
  offset? : Int = 0,
) -> Unit raise MaterializeError {
  self.source.decode_bf16_chunk(
    first_element,
    count,
    destination,
    destination_offset=offset,
  ) catch {
    error => raise Conversion(error)
  }
}
