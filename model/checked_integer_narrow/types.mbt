///|
pub(all) suberror CheckedIntegerNarrowError {
  InvalidGeometry
  InputByteCountMismatch(expected~ : Int32, actual~ : Int32)
  OutputByteBudgetExceeded
  InvalidStreamState
  StreamByteCountMismatch(expected~ : Int64, actual~ : Int64)
  ValueOutOfRange(row~ : Int32, column~ : Int32, value~ : Int64)
  DuplicateRowValue(
    row~ : Int32,
    first_column~ : Int32,
    second_column~ : Int32,
    value~ : Int64
  )
} derive(Debug, Eq)

///|
/// Explicitly releasable, little-endian I32 row-major table produced only
/// after every I64 source element passes range and per-row uniqueness checks.
pub struct CheckedI32Table {
  priv row_count : Int32
  priv column_count : Int32
  priv minimum_value : Int32
  priv exclusive_maximum_value : Int32
  priv mut bytes : FixedArray[Byte]?
  priv mut released : Bool
}

///|
pub extend CheckedIntegerNarrowError with Eq::{equal, not_equal}

///|
pub extend CheckedIntegerNarrowError with @debug.Debug::{to_repr}
