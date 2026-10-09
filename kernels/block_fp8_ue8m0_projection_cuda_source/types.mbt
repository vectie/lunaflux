///|
pub(all) suberror BlockFp8Ue8M0ProjectionCudaSourceError {
  InvalidGeometry
  InvalidSymbol
  SizeOverflow
} derive(Debug, Eq)

///|
pub struct BlockFp8Ue8M0ProjectionCudaSource {
  priv source : Bytes
  priv input_width : Int
  priv output_width : Int
  priv maximum_rows : Int
  priv block_width : Int
  priv function_symbol : String
}
