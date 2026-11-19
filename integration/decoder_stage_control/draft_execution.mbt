///|
/// Startup-bound draft effects. Prediction runs only over an already committed
/// prefix. poll includes retirement of the prepared output readback; token_at
/// reads its live host storage, not a synchronous device diagnostic. Token zero
/// is the previously emitted seed. No draft token is committed by this owner.
pub struct StageDraftExecution {
  priv capacity : Int32
  priv begin : () -> Unit raise
  priv begin_seeded : ((Int32) -> Unit raise)?
  priv poll : () -> Bool raise
  priv count : () -> Int32 raise
  priv token_at : (Int32) -> Int32 raise
}

///|
pub fn StageDraftExecution::new(
  maximum_tokens~ : Int32,
  begin~ : () -> Unit raise,
  poll~ : () -> Bool raise,
  count~ : () -> Int32 raise,
  token_at~ : (Int32) -> Int32 raise,
  begin_seeded? : (Int32) -> Unit raise,
) -> Self raise StageControlError {
  guard maximum_tokens > 0 else { raise InvalidConfiguration }
  { capacity: maximum_tokens, begin, begin_seeded, poll, count, token_at, }
}
