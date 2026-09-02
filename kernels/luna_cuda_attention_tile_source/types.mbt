///|
pub struct CudaPagedAttentionTileAbi {
  priv query_heads : Int
  priv key_value_heads : Int
  priv head_dimension : Int
  priv tokens_per_page : Int
  priv input_row_width : Int
  priv page_stride_values : Int64
  priv physical_page_capacity : Int64
  priv maximum_query_rows : Int
  priv maximum_query_tokens : Int
  priv maximum_page_entries : Int
} derive(Debug, Eq)

///|
pub struct CudaAttentionTileSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub struct CudaAttentionTileSource {
  priv lowering_digest : @luna_cuda_attention_tile_lowering.CudaAttentionTileLoweringDigest
  priv abi : CudaPagedAttentionTileAbi
  priv symbol : String
  priv bytes : Bytes
  priv digest : CudaAttentionTileSourceDigest
} derive(Debug, Eq)

///|
pub(all) enum CudaAttentionTileSourceRule {
  InvalidAbi
  InvalidSymbol
  UnsupportedLowering
  SourceLimit
} derive(Debug, Eq)

///|
pub(all) suberror CudaAttentionTileSourceError {
  InvalidCudaAttentionTileSource(CudaAttentionTileSourceRule)
} derive(Debug, Eq)
