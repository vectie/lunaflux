///|
/// Probability storage is part of numerical semantics, not an invisible
/// device scheduling choice. The online law rounds P before each block's P×V,
/// but keeps maxima, rescaling factors, denominators and accumulators in F32.
/// Sink mass is added to the final denominator without participating in maxima.
pub(all) enum AttentionProbabilityLaw {
  GlobalF32
  OrderedOnlineF32
  OnlineBf16(block_tokens~ : Int)
} derive(Eq, Debug)

///|
pub fn AttentionProbabilityLaw::validate(
  self : Self,
) -> Unit raise PrecisionPlanError {
  match self {
    GlobalF32 | OrderedOnlineF32 => ()
    OnlineBf16(block_tokens~) => {
      guard block_tokens > 0 &&
        block_tokens <= 4096 &&
        (block_tokens & (block_tokens - 1)) == 0 else {
        raise IllegalPlacement
      }
    }
  }
}
