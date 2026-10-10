///|
pub(all) suberror CheckedIntegerNarrowError {
  InvalidGeometry
  InputByteCountMismatch(expected~ : Int, actual~ : Int)
  OutputByteBudgetExceeded
  InvalidStreamState
  StreamByteCountMismatch(expected~ : Int64, actual~ : Int64)
  ValueOutOfRange(row~ : Int, column~ : Int, value~ : Int64)
  DuplicateRowValue(
    row~ : Int,
    first_column~ : Int,
    second_column~ : Int,
    value~ : Int64
  )
} derive(Debug, Eq)

///|
pub struct CheckedI32TableDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
/// Explicitly releasable, little-endian I32 row-major table produced only
/// after every I64 source element passes range and per-row uniqueness checks.
pub struct CheckedI32Table {
  priv row_count : Int
  priv column_count : Int
  priv minimum_value : Int
  priv exclusive_maximum_value : Int
  priv digest : CheckedI32TableDigest
  priv mut bytes : FixedArray[Byte]?
  priv mut released : Bool
}
