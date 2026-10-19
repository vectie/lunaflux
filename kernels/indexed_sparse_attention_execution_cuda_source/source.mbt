///|
/// CUDA lowering of a generic projected sparse attention plan. The arithmetic
/// remains the existing ordered BF16/F32 law. Attention distributes independent
/// query/head/score/value work. Index candidates retain each ordered dot once;
/// attention subsequently reuses its scratch under the ordered effect plan.
pub struct IndexedSparseAttentionCudaSource {
  priv plan : @precision.IndexedSparseAttentionPrecision
  priv symbols : ReadOnlyArray[String]
  priv source : Bytes
  priv order : ReadOnlyArray[Int]
  priv attention : @attention.SelectedSparseCudaSource
  priv index : @index_physical.RetainedIndexPlan
  priv score_tiles : @index_physical.CandidateScorePlan?
}

///|
pub fn indexed_sparse_attention_source(
  plan : @precision.IndexedSparseAttentionPrecision,
  symbol_prefix~ : String,
) -> IndexedSparseAttentionCudaSource raise {
  let (_, id, pool_size, _, _) = plan.index_geometry()
  let (qh, kvh, qkd, vd) = plan.attention_geometry()
  let (queries, history) = plan.capacity()
  let program = plan.program()
  let semantic_order = program
    .lower(Ordered)
    .commands(ComputeLane)
    .map(command => {
      match command {
        Execute(step~) =>
          match program.steps()[step].action() {
            Compute(stage) => stage
            _ => raise @precision.PrecisionPlanError::NoImplementation
          }
        _ => raise @precision.PrecisionPlanError::NoImplementation
      }
    })
  let symbols = [
    symbol_prefix + "_pool",
    symbol_prefix + "_index",
    symbol_prefix + "_attention_retained_scores_v1",
    symbol_prefix + "_index_scores",
  ]
  let pool = @pooling.dsa_key_pool_compression_bf16_source(
    head_dimension=id,
    pool_size~,
    maximum_tokens=history,
    function_symbol=symbols[0],
  )
  let index = @index_physical.RetainedIndexPlan::new(plan)
  let score_tiles = if plan.paged_history() is Some(_) {
    index.score_tiles(128)
  } else {
    None
  }
  let order = []
  for stage in semantic_order {
    if stage == 1 && score_tiles is Some(tiles) {
      let refinement = tiles.program()
      for command in refinement.lower(Ordered).commands(ComputeLane) {
        guard command is Execute(step~) &&
          refinement.steps()[step].action() is Compute(part) else {
          raise @precision.PrecisionPlanError::NoImplementation
        }
        order.push(if part == 0 { 3 } else { 1 })
      }
    } else {
      order.push(stage)
    }
  }
  let index_source = if plan.paged_history() is Some(_) {
    paged_index_source(plan, symbols[1], score_tiles, symbols[3])
  } else {
    retained_index_source(plan, index, symbols[1])
  }
  let attention = @attention.selected_sparse_source(
    query_heads=qh,
    kv_heads=kvh,
    qk_dimension=qkd,
    value_dimension=vd,
    selected_width=plan.selected_width(),
    maximum_history=history,
    maximum_queries=queries,
    symbol=symbols[2],
    published_slots=index.published_slots(),
    paged_history?=plan.paged_history(),
    accumulation=plan.attention_accumulation(),
  )
  {
    plan,
    order: ReadOnlyArray::from_array(order),
    attention,
    index,
    score_tiles,
    symbols: ReadOnlyArray::from_array(symbols),
    source: (if plan.paged_history() is Some(_) {
      paged_pool_source(plan, symbols[0])
    } else {
      pool.source_bytes()
    }) +
    b"\n" +
    index_source +
    b"\n" +
    attention.source_bytes(),
  }
}

///|
pub fn IndexedSparseAttentionCudaSource::scratch_bytes(self : Self) -> Int64 {
  Int64::max(self.attention.scratch_bytes(), self.index.scratch_bytes())
}

///|
/// IndexShare consumers execute only the final stage, not the index lifetime.
pub fn IndexedSparseAttentionCudaSource::attention_scratch_bytes(
  self : Self,
) -> Int64 {
  self.attention.scratch_bytes()
}

///|
pub fn IndexedSparseAttentionCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int) {
  if stage == 2 {
    self.attention.launch()
  } else if stage == 3 && self.score_tiles is Some(tiles) {
    (tiles.workgroups(), tiles.geometry().2)
  } else if stage == 1 {
    (self.index.geometry().0, 128)
  } else if self.plan.paged_history() is Some(_) {
    let updates = @index_physical.PoolUpdatePlan::new(
      rows=self.plan.capacity().0,
      pool=self.plan.index_geometry().2,
    ).unwrap()
    (updates.workgroups(), 128)
  } else {
    (1, 1)
  }
}

///|
pub fn IndexedSparseAttentionCudaSource::execution_order(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.order
}

///|
pub fn IndexedSparseAttentionCudaSource::plan(
  self : Self,
) -> @precision.IndexedSparseAttentionPrecision {
  self.plan
}

///|
pub fn IndexedSparseAttentionCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn IndexedSparseAttentionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
