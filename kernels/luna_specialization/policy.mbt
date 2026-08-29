///|
fn luna_specialization_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub struct LunaCompilerIdentity {
  priv toolchain_digest : String
  priv major : Int
  priv minor : Int
  priv patch : Int
} derive(Debug, Eq)

///|
pub fn LunaCompilerIdentity::new(
  toolchain_digest_sha256~ : String,
  major~ : Int,
  minor~ : Int,
  patch~ : Int,
) -> LunaCompilerIdentity raise LunaSpecializationError {
  if !luna_specialization_is_sha256(toolchain_digest_sha256) ||
    major < 0 ||
    minor < 0 ||
    patch < 0 {
    raise Invalid(CompilerIdentity)
  }
  { toolchain_digest: toolchain_digest_sha256, major, minor, patch }
}

///|
pub fn LunaCompilerIdentity::toolchain_digest_sha256(
  self : LunaCompilerIdentity,
) -> String {
  self.toolchain_digest
}

///|
pub fn LunaCompilerIdentity::major(self : LunaCompilerIdentity) -> Int {
  self.major
}

///|
pub fn LunaCompilerIdentity::minor(self : LunaCompilerIdentity) -> Int {
  self.minor
}

///|
pub fn LunaCompilerIdentity::patch(self : LunaCompilerIdentity) -> Int {
  self.patch
}

///|
pub struct LunaCompilerPolicy {
  priv identity : LunaCompilerIdentity
  priv optimization_level : Int
  priv fused_multiply_add : Bool
  priv reassociate : Bool
  priv max_registers : Int
} derive(Debug, Eq)

///|
pub fn LunaCompilerPolicy::new(
  identity~ : LunaCompilerIdentity,
  optimization_level~ : Int,
  fused_multiply_add~ : Bool,
  reassociate~ : Bool,
  max_registers~ : Int,
) -> LunaCompilerPolicy raise LunaSpecializationError {
  if optimization_level < 0 ||
    optimization_level > 3 ||
    max_registers <= 0 ||
    max_registers > 255 {
    raise Invalid(CompilerPolicy)
  }
  {
    identity,
    optimization_level,
    fused_multiply_add,
    reassociate,
    max_registers,
  }
}

///|
pub fn LunaCompilerPolicy::identity(
  self : LunaCompilerPolicy,
) -> LunaCompilerIdentity {
  self.identity
}

///|
pub fn LunaCompilerPolicy::optimization_level(self : LunaCompilerPolicy) -> Int {
  self.optimization_level
}

///|
pub fn LunaCompilerPolicy::fused_multiply_add(
  self : LunaCompilerPolicy,
) -> Bool {
  self.fused_multiply_add
}

///|
pub fn LunaCompilerPolicy::reassociate(self : LunaCompilerPolicy) -> Bool {
  self.reassociate
}

///|
pub fn LunaCompilerPolicy::max_registers(self : LunaCompilerPolicy) -> Int {
  self.max_registers
}

///|
pub(all) enum LunaNumericalPolicy {
  LunaBitExactOperationOrderV1
  LunaToleranceV1(
    maximum_absolute_error_ppb~ : Int64,
    maximum_relative_error_ppb~ : Int64,
    requires_token_agreement~ : Bool
  )
} derive(Debug, Eq)

///|
pub(all) enum LunaQuantizationPolicy {
  LunaUnquantizedBf16V1
} derive(Debug, Eq)

///|
pub struct LunaSpecializationEvidenceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaSpecializationEvidenceDigest::from_sha256(
  value : String,
) -> LunaSpecializationEvidenceDigest raise LunaSpecializationError {
  if !luna_specialization_is_sha256(value) {
    raise Invalid(Evidence)
  }
  LunaSpecializationEvidenceDigest(value)
}

///|
pub fn LunaSpecializationEvidenceDigest::as_hex(
  self : LunaSpecializationEvidenceDigest,
) -> String {
  self.0
}

///|
pub struct LunaSpecializationEvidence {
  priv scalar_referee : LunaSpecializationEvidenceDigest
  priv admitted_tensor_rows : LunaSpecializationEvidenceDigest
  priv dispatch_canary : LunaSpecializationEvidenceDigest
  priv benchmark : LunaSpecializationEvidenceDigest
} derive(Debug, Eq)

///|
pub fn LunaSpecializationEvidence::new(
  scalar_referee~ : LunaSpecializationEvidenceDigest,
  admitted_tensor_rows~ : LunaSpecializationEvidenceDigest,
  dispatch_canary~ : LunaSpecializationEvidenceDigest,
  benchmark~ : LunaSpecializationEvidenceDigest,
) -> LunaSpecializationEvidence {
  { scalar_referee, admitted_tensor_rows, dispatch_canary, benchmark }
}

///|
pub fn LunaSpecializationEvidence::scalar_referee(
  self : LunaSpecializationEvidence,
) -> LunaSpecializationEvidenceDigest {
  self.scalar_referee
}

///|
pub fn LunaSpecializationEvidence::admitted_tensor_rows(
  self : LunaSpecializationEvidence,
) -> LunaSpecializationEvidenceDigest {
  self.admitted_tensor_rows
}

///|
pub fn LunaSpecializationEvidence::dispatch_canary(
  self : LunaSpecializationEvidence,
) -> LunaSpecializationEvidenceDigest {
  self.dispatch_canary
}

///|
pub fn LunaSpecializationEvidence::benchmark(
  self : LunaSpecializationEvidence,
) -> LunaSpecializationEvidenceDigest {
  self.benchmark
}

///|
pub struct LunaSpecializationLimits {
  priv max_canonical_bytes : Int
  priv max_bound_regions : Int
} derive(Debug, Eq)

///|
pub fn LunaSpecializationLimits::new(
  max_canonical_bytes~ : Int,
  max_bound_regions~ : Int,
) -> LunaSpecializationLimits raise LunaSpecializationError {
  if max_canonical_bytes <= 0 || max_bound_regions <= 0 {
    raise Invalid(CanonicalBytes)
  }
  { max_canonical_bytes, max_bound_regions }
}
