///|
const LUNA_TILE_PROMOTION_MAX_CANONICAL_BYTES : Int = 262144

///|
fn luna_tile_promotion_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for character in value {
    if !(character is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub struct LunaTileOptimizerDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileOptimizerDigest::from_sha256(
  value : String,
) -> LunaTileOptimizerDigest raise LunaTileOptimizerPromotionError {
  if !luna_tile_promotion_is_sha256(value) {
    raise LunaTileOptimizerPromotionInvalid(Digest)
  }
  LunaTileOptimizerDigest(value)
}

///|
pub fn LunaTileOptimizerDigest::as_hex(
  self : LunaTileOptimizerDigest,
) -> String {
  self.0
}

///|
fn luna_tile_promotion_digest(bytes : Bytes) -> LunaTileOptimizerDigest {
  LunaTileOptimizerDigest(@crypto.bytes_to_hex_string(@crypto.sha256(bytes)))
}

///|
pub(all) enum LunaTileOptimizerPath {
  LunaTileParallelSimtV1
  LunaTileTensorCoreBf16Mma16x16x16V1
} derive(Debug, Eq)

///|
fn LunaTileOptimizerPath::stable_name(self : LunaTileOptimizerPath) -> String {
  match self {
    LunaTileParallelSimtV1 => "parallel-simt-v1"
    LunaTileTensorCoreBf16Mma16x16x16V1 => "tensor-core-bf16-m16n16k16-v1"
  }
}

///|
pub struct LunaTilePromotionDeploymentIdentity(LunaTileOptimizerDigest) derive (
  Debug,
  Eq,
)

///|
pub fn LunaTilePromotionDeploymentIdentity::from_sha256(
  value : String,
) -> LunaTilePromotionDeploymentIdentity raise LunaTileOptimizerPromotionError {
  LunaTilePromotionDeploymentIdentity(
    LunaTileOptimizerDigest::from_sha256(value),
  )
}

///|
pub fn LunaTilePromotionDeploymentIdentity::as_hex(
  self : LunaTilePromotionDeploymentIdentity,
) -> String {
  self.0.as_hex()
}

///|
pub struct AdmittedLunaTileSimtPhysicalEvidence {
  priv candidate : @luna_tile_ir.LunaTileParallelDigest
  priv source : @luna_tile_ir.LunaTileCudaSourceDigest
  priv serial : @luna_tile_ir.LunaTileCudaTranslationDigest
  priv module_digest : LunaTileOptimizerDigest
  priv serial_module_digest : LunaTileOptimizerDigest
  priv sealed_result : LunaTileOptimizerDigest
  priv canonical : Bytes
  priv digest : LunaTileOptimizerDigest
}

///|
pub struct LunaTilePairedBenchmarkObservation {
  priv candidate_time_ns : Int64
  priv fallback_time_ns : Int64
} derive(Debug, Eq)

///|
pub struct AdmittedLunaTilePairedBenchmarkWin {
  priv path : LunaTileOptimizerPath
  priv candidate : LunaTileOptimizerDigest
  priv fallback : LunaTileOptimizerDigest
  priv physical : LunaTileOptimizerDigest
  priv observation_count : Int
  priv candidate_total_ns : Int64
  priv fallback_total_ns : Int64
  priv observed_win_ppm : Int
  priv canonical : Bytes
  priv digest : LunaTileOptimizerDigest
}

///|
pub struct LunaTilePromotionArtifactInput {
  priv module_digest : LunaTileOptimizerDigest
  priv entry_point : String
}

///|
pub fn LunaTilePromotionArtifactInput::new(
  module_digest~ : LunaTileOptimizerDigest,
  entry_point~ : String,
) -> LunaTilePromotionArtifactInput raise LunaTileOptimizerPromotionError {
  if entry_point.is_empty() || entry_point.length() > 255 {
    raise LunaTileOptimizerPromotionInvalid(Artifact)
  }
  for character in entry_point {
    if !(character is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) {
      raise LunaTileOptimizerPromotionInvalid(Artifact)
    }
  }
  { module_digest, entry_point }
}

///|
pub struct LunaTilePromotionApprovalRecord {
  priv path : LunaTileOptimizerPath
  priv deployment : LunaTilePromotionDeploymentIdentity
  priv promotion_epoch : UInt64
  priv candidate : LunaTileOptimizerDigest
  priv source : LunaTileOptimizerDigest
  priv fallback : LunaTileOptimizerDigest
  priv module_digest : LunaTileOptimizerDigest
  priv entry_point : String
  priv physical : LunaTileOptimizerDigest
  priv benchmark : LunaTileOptimizerDigest
  priv canonical : Bytes
  priv digest : LunaTileOptimizerDigest
}

///|
pub struct ApprovedLunaTileOptimizerCatalogEntry {
  priv record : LunaTilePromotionApprovalRecord
  priv approval : @luna_capability_manifest.LunaAdmittedExternalRecord
}

///|
pub(all) enum LunaTileStartupSelection {
  LunaTileBaselineSelected
  LunaTileApprovedOptimizerSelected(ApprovedLunaTileOptimizerCatalogEntry)
}
