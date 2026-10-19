///|
/// Expanded and latent execution have different BF16 rounding boundaries.
/// Latent execution explicitly permits reassociating the key/value projection
/// around attention, with matrix F32 reductions and BF16 publication at each
/// new boundary. It is not an exact rewrite of the expanded program.
pub(all) enum SparseCacheLaw {
  ExpandedBFloat16
  LatentBFloat16
} derive(Debug, Eq)

///|
pub extend SparseCacheLaw with Eq::{equal, not_equal}

///|
pub extend SparseCacheLaw with @debug.Debug::{to_repr}

///|
pub fn IndexedSparseAttentionPrecision::latent_cache(
  self : Self,
  rank : Int,
) -> IndexedSparseAttentionPrecision raise PrecisionPlanError {
  IndexedSparseAttentionPrecision::new(
    index_head_count=self.index_heads,
    index_head_dimension=self.index_dimension,
    query_head_count=self.query_heads,
    key_value_head_count=1,
    query_key_dimension=rank,
    value_dimension=rank,
    pool_size=self.pool_size,
    top_k=self.top_k,
    append_incomplete_tail=self.tail,
    maximum_queries=self.queries,
    maximum_history=self.history,
    maximum_workspace_bytes=self.workspace,
    paged_history?=self.paging,
    attention_accumulation=self.accumulation,
    index_scoring=self.scoring,
    score_dimension=self.score_dimension,
    shared_key_value=true,
  )
}
