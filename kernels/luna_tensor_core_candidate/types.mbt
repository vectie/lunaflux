///|
pub struct LunaTileTensorCoreSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileTensorCoreSourceDigest::as_hex(
  self : LunaTileTensorCoreSourceDigest,
) -> String {
  self.0
}

///|
pub struct LunaTileTensorCoreCandidateDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileTensorCoreCandidateDigest::as_hex(
  self : LunaTileTensorCoreCandidateDigest,
) -> String {
  self.0
}

///|
/// Inert exact-sm120 WMMA source candidate. It has no compiler, device,
/// manifest, dispatch, performance, SASS-qualification, or promotion authority.
pub struct LunaTileBf16Mma16x16x16Candidate {
  priv program_digest : @luna_tile_ir.LunaTileProgramDigest
  priv serial_oracle_digest : @luna_tile_ir.LunaTileCudaTranslationDigest
  priv parallel_plan_digest : @luna_tile_ir.LunaTileParallelDigest
  priv tile_count : Int32
  priv entry_point : String
  priv source : Bytes
  priv source_digest : LunaTileTensorCoreSourceDigest
  priv canonical : Bytes
  priv digest : LunaTileTensorCoreCandidateDigest
} derive(Debug, Eq)

///|
pub extend LunaTileTensorCoreSourceDigest with @debug.Debug::{to_repr}

///|
pub extend LunaTileTensorCoreSourceDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorCoreSourceDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileTensorCoreCandidateDigest with @debug.Debug::{to_repr}

///|
pub extend LunaTileTensorCoreCandidateDigest with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorCoreCandidateDigest with Hash::{hash, hash_combine}

///|
pub extend LunaTileBf16Mma16x16x16Candidate with @debug.Debug::{to_repr}

///|
pub extend LunaTileBf16Mma16x16x16Candidate with Eq::{not_equal, equal}
