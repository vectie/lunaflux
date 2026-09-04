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
pub struct CudaPartitionedAttentionWorkspace {
  priv partial_state_count : Int64
  priv maximum_offset_bytes : Int64
  priv denominator_offset_bytes : Int64
  priv numerator_offset_bytes : Int64
  priv total_bytes : Int64
} derive(Debug, Eq)

///|
/// Deterministic two-entry-point source bundle for a partitioned online
/// softmax reduction. The first entry point computes independent KV
/// partitions; the second applies the associative merge law.
pub struct CudaPartitionedAttentionTileSource {
  priv lowering_digest : @luna_cuda_attention_tile_lowering.CudaAttentionTileLoweringDigest
  priv abi : CudaPagedAttentionTileAbi
  priv partial_symbol : String
  priv merge_symbol : String
  priv workspace : CudaPartitionedAttentionWorkspace
  priv bytes : Bytes
  priv digest : CudaAttentionTileSourceDigest
} derive(Debug, Eq)

///|
pub(all) enum CudaAttentionTileSourceRule {
  InvalidAbi
  InvalidSymbol
  UnsupportedLowering
  WorkspaceLimit
  SourceLimit
} derive(Debug, Eq)

///|
pub(all) suberror CudaAttentionTileSourceError {
  InvalidCudaAttentionTileSource(CudaAttentionTileSourceRule)
} derive(Debug, Eq)
