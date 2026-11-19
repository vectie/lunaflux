///|
pub extend IndexedParameterBankLayoutError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend IndexedParameterBankLayoutError with Eq::{not_equal, equal}

///|
pub extend IndexedParameterBankLayoutDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend IndexedParameterBankLayoutDigest with Eq::{not_equal, equal}

///|
pub extend IndexedParameterBankLayoutDigest with Hash::{hash, hash_combine}
