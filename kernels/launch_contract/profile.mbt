///|
pub struct KernelProfileId(Int) derive(Debug, Eq)

///|
/// Exact full-prefill shape understood by an AOT launch contract.
///
/// `token_rows` is checked rather than inferred by a kernel. This prevents an
/// entry point admitted for the plan envelope from being reused silently for
/// a smaller live profile.
pub struct KernelProfile {
  id : KernelProfileId
  batch_rows : Int
  sequence_tokens : Int
  token_rows : Int64
} derive(Debug, Eq)

///|
pub fn KernelProfileId::from_stable_int(
  value : Int,
) -> KernelProfileId raise LaunchContractError {
  if value <= 0 {
    raise InvalidProfile(NonPositiveId)
  }
  KernelProfileId(value)
}

///|
pub fn KernelProfileId::as_int(self : KernelProfileId) -> Int {
  self.0
}

///|
pub fn KernelProfile::full_prefill(
  id~ : KernelProfileId,
  batch_rows~ : Int,
  sequence_tokens~ : Int,
  token_rows~ : Int64,
) -> KernelProfile raise LaunchContractError {
  if batch_rows <= 0 || sequence_tokens <= 0 || token_rows <= 0L {
    raise InvalidProfile(NonPositiveDimension)
  }
  let batch = batch_rows.to_int64()
  let sequence = sequence_tokens.to_int64()
  if batch > 9223372036854775807L / sequence {
    raise InvalidProfile(TokenRowOverflow)
  }
  if batch * sequence != token_rows {
    raise InvalidProfile(TokenRowMismatch)
  }
  { id, batch_rows, sequence_tokens, token_rows }
}

///|
pub fn KernelProfile::id(self : KernelProfile) -> KernelProfileId {
  self.id
}

///|
pub fn KernelProfile::batch_rows(self : KernelProfile) -> Int {
  self.batch_rows
}

///|
pub fn KernelProfile::sequence_tokens(self : KernelProfile) -> Int {
  self.sequence_tokens
}

///|
pub fn KernelProfile::token_rows(self : KernelProfile) -> Int64 {
  self.token_rows
}
