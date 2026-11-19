///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceDigest with @debug.Debug::{
  to_repr,
}

///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceDigest with Eq::{
  not_equal,
  equal,
}

///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceDigest with Hash::{
  hash,
  hash_combine,
}

///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceRule with @debug.Debug::{
  to_repr,
}

///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceRule with Eq::{
  not_equal,
  equal,
}

///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceError with @debug.Debug::{
  to_repr,
}

///|
pub extend PagedAttentionReadOnlyPhysicalEvidenceError with Eq::{
  not_equal,
  equal,
}
