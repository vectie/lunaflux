///|
test "export stable selection edge probe" {
  if @env.get_env_var("LUNA_SELECTION_EDGE_EXPORT") == Some("1") {
    println("LUNA_SELECTION_BEGIN")
    for candidates in [16, 1024] {
      let semantic = @precision.IndexedSparseAttentionPrecision::new(
        index_head_count=1,
        index_head_dimension=4,
        query_head_count=1,
        key_value_head_count=1,
        query_key_dimension=4,
        value_dimension=4,
        pool_size=1,
        top_k=4,
        append_incomplete_tail=false,
        maximum_queries=1,
        maximum_history=candidates,
        maximum_workspace_bytes=1000000L,
      )
      let plan = @index_physical.RetainedIndexPlan::new(semantic)
      let selection = stable_selection_source(plan, 1, "limit", "best")
      let source =
        $|extern "C" __global__ void selection_edge_\{candidates}(const float *scores,int *available,int *selected_indices,int candidates) {
        $| const int output_base=0,limit=4;
        $| if(threadIdx.x<4)selected_indices[threadIdx.x]=-1;
        $| __syncthreads();
        $| \{selection}
        $|}
      println(source)
    }
    println("LUNA_SELECTION_END")
  }
}
