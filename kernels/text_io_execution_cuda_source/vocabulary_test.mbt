///|
async test "export independent vocabulary shard and stable join" {
  let text = StringBuilder()
  for index, (vocabulary, hidden, outputs) in [(17, 64, 3), (154880, 4096, 1)] {
    let plan = @precision.TextIoPrecision::new(
      vocabulary_size=vocabulary,
      hidden_width=hidden,
      stream_count=4,
      maximum_rows=64,
      maximum_output_rows=outputs,
      normalization_epsilon=1.0e-5,
      maximum_workspace_bytes=10000000L,
      head_law=ShapeAwareFloat32,
    )
    let full = @text_io_execution_cuda_source.text_io_source(
      plan,
      symbol_prefix="full\{index}",
    )
    text.write_string(@utf8.decode(full.source_bytes()))
    for rank in 0..<2 {
      let shard = @parallel.VocabularyShard::new(
        vocabulary~,
        rank~,
        world=2,
        maximum_outputs=outputs,
      )
      let source = @text_io_execution_cuda_source.text_io_source(
        plan,
        symbol_prefix="shard\{index}_\{rank}",
        vocabulary_shard=shard,
      )
      assert_eq(
        source.head_bytes(),
        shard.columns().1.to_int64() * hidden.to_int64() * 2L,
      )
      assert_eq(
        source.local_logits_bytes(),
        shard.columns().1.to_int64() * outputs.to_int64() * 4L,
      )
      text.write_string(@utf8.decode(source.source_bytes()))
    }
    let split = vocabulary / 2
    text <+
      "\nvoid head\{index}(bool shard,int rank,int *count,__nv_bfloat16 *x,__nv_bfloat16 *w,float *y) { if(!shard) full\{index}_head<<<\{full.head_launch().0},128>>>(count,x,w,y); else if(rank==0) shard\{index}_0_head<<<\{(split+3)/4},128>>>(count,x,w,y); else shard\{index}_1_head<<<\{(vocabulary-split+3)/4},128>>>(count,x,w+(unsigned long long)\{split}*\{hidden},y); }\n"
    text <+
      "void greedy\{index}(int rank,int *counts,int *count,int *selected,float *logits,float *candidates,int *tokens) { if(rank<0) full\{index}_greedy<<<\{outputs},256>>>(counts,count,selected,logits,tokens); else if(rank==0) shard\{index}_0_greedy<<<\{outputs},256>>>(counts,count,selected,logits,candidates); else shard\{index}_1_greedy<<<\{outputs},256>>>(counts,count,selected,logits,candidates); }\n"
    text <+
      "void join\{index}(int *count,float *all,int *tokens) {shard\{index}_0_greedy_join<<<\{outputs},1>>>(count,all,tokens);}\n"
  }
  if @env.get_env_var("LUNA_VOCAB_TP_HEADER") is Some(path) {
    @fs.write_file(path, text.to_string(), create_mode=CreateNew)
  }
}
