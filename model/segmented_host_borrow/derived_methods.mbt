///|
pub extend SegmentedHostBorrowError with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend SegmentedHostBorrowError with Eq::{not_equal, equal}
