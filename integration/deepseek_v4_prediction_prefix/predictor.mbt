///|
/// Complete local DSpark prediction dataflow, borrowing ordered base captures,
/// embedding/vocabulary tables and explicit result/RNG ports. This owner does
/// not verify or commit drafts; the base scheduler retains that authority.
pub struct DeepSeekPredictor {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv main_rows : Int
  priv main : DeepSeekPredictionMain
  priv seed : @text_frame.SeededTokenBlockFrame
  priv embedding : @text_frame.TextIoFrame
  priv blocks : Array[DeepSeekPredictionBlock]
  priv head : DeepSeekPredictionHead
  priv buffers : Array[@device.Allocation]
  priv prime : Array[@device.OrderedKernelLaunch]
  priv predict : Array[@device.OrderedKernelLaunch]
  priv mut state : AttentionState
}

///|
pub fn DeepSeekPredictor::new(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_main_rows~ : Int,
  maximum_positions~ : Int,
  maximum_workspace_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  temperature~ : Double,
  accuracy~ : @precision.Accuracy,
) -> Self raise {
  let rows = spec.dspark_block_size()
  require_prediction_stage(spec, spec.prediction_stage_count() - 1)
  let embedding_plan = @precision.TextIoPrecision::new(
    vocabulary_size=spec.vocabulary_size(),
    hidden_width=spec.hidden_size(),
    stream_count=spec.hyper_connection_width(),
    maximum_rows=rows,
    maximum_output_rows=1,
    normalization_epsilon=spec.rms_norm_epsilon(),
    maximum_workspace_bytes=0L,
    boundary=Ingress,
  )
  {
    spec,
    main_rows: maximum_main_rows,
    main: DeepSeekPredictionMain::new(
      spec,
      maximum_rows=maximum_main_rows,
      maximum_workspace_bytes~,
    ),
    seed: @text_frame.SeededTokenBlockFrame::new(
      @precision.SeededTokenBlockPrecision::new(
        vocabulary_size=spec.vocabulary_size(),
        block_rows=rows,
        maximum_main_rows~,
        noise_token_id=spec.dspark_noise_token_id(),
      ),
      symbol_prefix="deepseek_prediction_seed",
    ),
    embedding: @text_frame.TextIoFrame::new(
      @text_source.text_io_source(
        embedding_plan,
        symbol_prefix="deepseek_prediction_embedding",
      ),
      boundary=Ingress,
    ),
    blocks: Array::makei(spec.prediction_stage_count(), stage => {
      DeepSeekPredictionBlock::new(
        spec,
        stage~,
        maximum_main_rows~,
        maximum_positions~,
        maximum_workspace_bytes~,
        maximum_expert_bytes~,
        accuracy~,
      )
    }),
    head: DeepSeekPredictionHead::new(
      spec,
      temperature~,
      maximum_workspace_bytes~,
    ),
    buffers: [],
    prime: [],
    predict: [],
    state: Fresh,
  }
}

///|
pub fn DeepSeekPredictor::source_bytes(self : Self) -> Bytes raise {
  let source = StringBuilder()
  for
    code in [
      self.main.source_bytes(),
      self.seed.source_bytes(),
      self.embedding.source_bytes(),
    ] {
    source.write_string(@utf8.decode(code))
    source.write_string("\n")
  }
  for block in self.blocks {
    source.write_string(@utf8.decode(block.source_bytes()))
    source.write_string("\n")
  }
  source.write_string(@utf8.decode(self.head.source_bytes()))
  @utf8.encode(source.to_string())
}

///|
fn DeepSeekPredictor::residual_bytes(self : Self) -> Int64 {
  self.spec.dspark_block_size().to_int64() *
  self.spec.hyper_connection_width().to_int64() *
  self.spec.hidden_size().to_int64() *
  2L
}

///|
/// Exact private footprint, excluding caller-owned capture, tables and ports.
/// Resolves all three blocks and the head before any checkpoint upload begins.
pub fn DeepSeekPredictor::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.blocks.fold(
    init=self.residual_bytes() * 2L +
      self.seed.workspace_bytes() +
      self.main.required_device_bytes(checkpoint) +
      self.head.required_device_bytes(checkpoint),
    (total, block) => total + block.required_device_bytes(checkpoint),
  )
}

///|
pub fn DeepSeekPredictor::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  main_counts~ : @device.Allocation,
  main_positions~ : @device.Allocation,
  captured_hidden~ : @device.Allocation,
  target_layers~ : ArrayView[Int],
  embedding_weight~ : @device.Allocation,
  vocabulary_weight~ : @device.Allocation,
  seed_token~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  start_position~ : @device.Allocation,
  tokens~ : @device.Allocation,
  confidence~ : @device.Allocation,
  rng~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    target_layers == self.spec.dspark_target_layer_ids()[:] &&
    chunk_bytes > 0 else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let rows = self.spec.dspark_block_size()
  // All caller-owned spans are checked before the first private allocation.
  for
    (b, bytes, alignment) in [
      (main_counts, 20L, 4L),
      (main_positions, self.main_rows.to_int64() * 4L, 4L),
      (
        captured_hidden,
        self.main_rows.to_int64() *
        self.spec.hidden_size().to_int64() *
        self.spec.dspark_target_layer_ids().length().to_int64() *
        2L,
        2L,
      ),
      (
        embedding_weight,
        self.spec.vocabulary_size().to_int64() *
        self.spec.hidden_size().to_int64() *
        2L,
        2L,
      ),
      (
        vocabulary_weight,
        self.spec.vocabulary_size().to_int64() *
        self.spec.hidden_size().to_int64() *
        2L,
        2L,
      ),
      (tokens, (rows.to_int64() + 1L) * 4L, 4L),
      (confidence, rows.to_int64() * 4L, 4L),
      (rng, 16L, 8L),
      (seed_token, 4L, 4L),
      (reset_flag, 4L, 4L),
      (start_position, 4L, 4L),
    ] {
    context.validate_allocation_region(
      b,
      offset=0L,
      byte_count=bytes,
      alignment~,
    )
  }
  for result in [tokens, confidence] {
    for
      read_port in [
        main_counts, main_positions, captured_hidden, embedding_weight, vocabulary_weight,
        seed_token, reset_flag, start_position, rng,
      ] {
      guard !result.same_storage(read_port) else {
        raise @precision.PrecisionPlanError::IllegalPlacement
      }
    }
  }
  guard !tokens.same_storage(confidence) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.state = Failed
  for _ in 0..<2 {
    self.buffers.push(context.allocate(self.residual_bytes()))
  }
  self.main.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts=main_counts,
    captured_hidden~,
    target_layers~,
    maximum_device_bytes=self.main.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.seed.prepare(
    context,
    aot,
    main_counts~,
    start_position~,
    seed_token~,
    output_chain=tokens,
    maximum_workspace_bytes=self.seed.workspace_bytes(),
  )
  self.embedding.prepare_ingress(
    aot,
    counts=self.seed.counts(),
    token_ids=self.seed.token_ids(),
    embedding=embedding_weight,
    input_residual=self.buffers[0],
  )
  for stage, block in self.blocks {
    block.prepare(
      context,
      aot,
      checkpoint,
      root,
      main_counts~,
      draft_counts=self.seed.counts(),
      main_positions~,
      draft_positions=self.seed.positions(),
      main_hidden=self.main.output(),
      input=self.buffers[stage % 2],
      output=self.buffers[(stage + 1) % 2],
      reset_flag~,
      start_position~,
      maximum_device_bytes=block.required_device_bytes(checkpoint),
      chunk_bytes~,
      limits~,
    )
  }
  self.head.prepare(
    context,
    aot,
    checkpoint,
    root,
    counts=self.seed.counts(),
    residual=self.buffers[self.blocks.length() % 2],
    vocabulary_weight~,
    tokens~,
    confidence~,
    rng~,
    maximum_device_bytes=self.head.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.prime.append(self.main.launches())
  self.predict.append(self.main.launches())
  self.predict.append(self.seed.launches())
  self.predict.append(self.embedding.prefix())
  for block in self.blocks {
    self.prime.append(block.prime_launches())
    self.predict.append(block.prediction_launches())
  }
  self.predict.append(self.head.launches())
  self.state = Prepared
}

///|
pub fn DeepSeekPredictor::prime_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.prime[:]
}

///|
pub fn DeepSeekPredictor::prediction_launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.predict[:]
}

///|
pub fn DeepSeekPredictor::logits(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.head.logits()
}

///|
pub fn DeepSeekPredictor::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.prime.clear()
  self.predict.clear()
  self.head.close()
  for i in 0..<self.blocks.length() {
    self.blocks[self.blocks.length() - 1 - i].close()
  }
  self.embedding.close()
  self.seed.close()
  self.main.close()
  while self.buffers.last() is Some(b) {
    b.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
