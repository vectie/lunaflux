///|
pub(all) suberror BlockFp8Ue8M0ProjectionCudaSourceError {
  InvalidGeometry
  InvalidSymbol
  SizeOverflow
  UnsupportedArithmetic
} derive(Debug, Eq)

///|
pub struct BlockFp8Ue8M0ProjectionCudaSource {
  priv source : Bytes
  priv input_width : Int32
  priv output_width : Int32
  priv maximum_rows : Int32
  priv block_width : Int32
  priv function_symbol : String
  priv matrix : Bool
}
