///|
pub extend Bf16BinaryCudaSourceError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend Bf16BinaryCudaSourceError with Eq::{not_equal, equal}
