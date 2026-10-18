///|
test "full vocabulary publication keeps head arithmetic and exposes logit transport" {
  let plan = @precision.TextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=64,
    stream_count=4,
    maximum_rows=8,
    maximum_output_rows=3,
    normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=100000L,
    head_law=ShapeAwareFloat32,
  )
  let shard = @parallel.VocabularyShard::new(
    vocabulary=17,
    rank=0,
    world=3,
    maximum_outputs=3,
    publication=FullLogits,
  )
  let source = @text_io_execution_cuda_source.text_io_source(
    plan,
    symbol_prefix="logits",
    vocabulary_shard=shard,
  )
  assert_eq(source.symbol(3), "logits_logits_pack")
  assert_eq(source.symbol(4), "logits_logits_join")
  assert_eq(
    source.workspace_bytes(),
    plan.normalized_bytes() + 3L * 5L * 4L + 18L * 4L * 4L + 3L * 17L * 4L,
  )
  let code = @utf8.decode(source.source_bytes())
  assert_true(code.contains("logits_logits_pack"))
  assert_true(code.contains("logits_logits_join"))
  assert_false(code.contains("_greedy_join"))
  assert_true(code.contains("word<end-first"))
}

///|
async test "export actual uneven full-logit transport" {
  guard @env.get_env_var("LUNA_VOCAB_LOGITS_HEADER") is Some(path) else {
    return
  }
  let text = StringBuilder()
  let plan = @precision.TextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=64,
    stream_count=4,
    maximum_rows=8,
    maximum_output_rows=3,
    normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=100000L,
    head_law=ShapeAwareFloat32,
  )
  for rank in 0..<3 {
    let shard = @parallel.VocabularyShard::new(
      vocabulary=17,
      rank~,
      world=3,
      maximum_outputs=3,
      publication=FullLogits,
    )
    let source = @text_io_execution_cuda_source.text_io_source(
      plan,
      symbol_prefix="shard\{rank}",
      vocabulary_shard=shard,
    )
    text.write_string(@utf8.decode(source.source_bytes()))
    text.write_string("\n")
  }
  text <+
    "void pack(int rank,int *c,int *n,int *selected,float *local,float *out){if(rank==0)shard0_logits_pack<<<3,256>>>(c,n,selected,local,out);else if(rank==1)shard1_logits_pack<<<3,256>>>(c,n,selected,local,out);else shard2_logits_pack<<<3,256>>>(c,n,selected,local,out);}\n"
  text <+
    "void join(int *n,float *all,float *out){shard0_logits_join<<<3,256>>>(n,all,out);}\n"
  @fs.write_file(path, text.to_string(), create_mode=CreateNew)
}
