///|
/// Prepared model stage: owns text weights only at model boundaries, connected
/// decoder layers, and distinct input/output residuals. Request ports/module/
/// stream are borrowed. A complete range can bind worker delivery; a partial
/// stage exposes residuals for a composed pipeline, never a fake model token.
pub struct GlmDecoderStage {
  priv range : @pipeline.DecoderStageRange
  priv decoder : GlmDecoderSlice
  priv text : GlmTextIo
  priv residual_bytes : Int64
  priv maximum_rows : Int
  priv maximum_history : Int
  priv vocabulary : Int
  priv maximum_sequences : Int
  priv maximum_outputs : Int
  priv residuals : Array[@device.Allocation]
  priv mut counts : @device.Allocation?
  priv mut activation_leased : Bool
  priv mut active_rows : Int
  priv mut submission_epoch : UInt64
  priv mut status : RecurrentLayerState
}

///|
pub fn GlmDecoderStage::new(
  spec : @glm53_spec.Glm53FlashSpec,
  range : @pipeline.DecoderStageRange,
  maximum_rows~ : Int,
  maximum_history~ : Int,
  index_layer_norm_epsilon~ : Double,
  maximum_layer_state_bytes~ : Int64,
  maximum_layer_workspace_bytes~ : Int64,
  maximum_layer_expert_bytes~ : Int64,
  maximum_text_workspace_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
  hc_reduction? : @precision_ir.HyperControlReduction = Ordered,
  recurrent_evaluation? : @precision_ir.RecurrentEvaluationLaw = OrderedTokenFloat32,
  matrix_law? : @precision_ir.DenseDotLaw = ScalarFloat32,
  matrix_staging? : @matrix.MatrixStaging = Synchronous,
  matrix_tile_rows? : Int = 16,
  sparse_matrix_staging? : @matrix.MatrixStaging = Synchronous,
  sparse_accumulation? : @precision_ir.AttentionAccumulationLaw = OrderedFloat32,
  index_scoring? : @precision_ir.IndexScoringLaw = OrderedBFloat16,
  sparse_cache? : @precision_ir.SparseCacheLaw = ExpandedBFloat16,
  expert_scaled_dot_law? : @precision_ir.ScaledDotReductionLaw = DecodeOperandsFloat32,
  expert_transport? : @matrix.ExpertTransport = IndexedRows,
  expert_fragment_slots? : Int = 1,
  expert_scale_partitions? : Int = 1,
  moe_reduction? : @precision_ir.MoeReductionPlacement = ExpertPartialsFloat32,
  tensor_publication? : @precision_ir.ProjectionPublication = ReduceFloat32BeforeRound,
  shared_matrix_columns? : Int = 64,
  tensor_group? : (Int, Int),
  shard_vocabulary? : Bool = false,
  vocabulary_publication? : @parallel.VocabularyPublication = GreedyCandidates,
  maximum_sequences? : Int = 1,
  cache_slots? : Int = 1,
  maximum_output_rows? : Int = 1,
  paged_history? : @history_domain.PagedHistoryDomain,
) -> Self raise {
  guard range.layer_count() == spec.layer_count() &&
    maximum_history >= maximum_rows &&
    maximum_sequences > 0 &&
    maximum_sequences <= maximum_rows &&
    cache_slots >= maximum_sequences &&
    maximum_output_rows > 0 &&
    maximum_output_rows <= maximum_sequences else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  guard vocabulary_publication == GreedyCandidates || shard_vocabulary else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let (first_layer, end_layer) = range.layer_range()
  // TP workers receive identical token IDs. Only the last rank owns model
  // egress; computing a discarded full vocabulary on every rank is redundant.
  let boundary : @precision_ir.TextBoundary = if !shard_vocabulary &&
    tensor_group is Some((rank, world)) &&
    rank != world - 1 &&
    range.text_boundary().has_egress() {
    if range.text_boundary().has_ingress() {
      Ingress
    } else {
      Interior
    }
  } else {
    range.text_boundary()
  }
  let vocabulary_shard = if shard_vocabulary {
    guard tensor_group is Some((rank, world)) &&
      range.text_boundary().has_egress() else {
      raise @precision_ir.PrecisionPlanError::IllegalPlacement
    }
    Some(
      @parallel.VocabularyShard::new(
        vocabulary=spec.vocabulary_size(),
        rank~,
        world~,
        maximum_outputs=maximum_output_rows,
        publication=vocabulary_publication,
      ),
    )
  } else {
    None
  }
  let text_plan = flash_text_io_plan(
    spec,
    maximum_rows~,
    maximum_output_rows~,
    maximum_workspace_bytes=maximum_text_workspace_bytes,
    boundary~,
    head_law=if matrix_law == ShapeAwareFloat32 {
      ShapeAwareFloat32
    } else {
      ScalarFloat32
    },
  )
  {
    range,
    decoder: GlmDecoderSlice::new(
      spec,
      hc_reduction~,
      first_layer~,
      end_layer~,
      maximum_rows~,
      maximum_sequences~,
      cache_slots~,
      maximum_history~,
      paged_history?,
      index_layer_norm_epsilon~,
      maximum_layer_state_bytes~,
      maximum_layer_workspace_bytes~,
      maximum_layer_expert_bytes~,
      accuracy~,
      recurrent_evaluation~,
      matrix_law~,
      matrix_staging~,
      matrix_tile_rows~,
      sparse_matrix_staging~,
      sparse_accumulation~,
      index_scoring~,
      sparse_cache~,
      expert_scaled_dot_law~,
      expert_transport~,
      expert_fragment_slots~,
      expert_scale_partitions~,
      moe_reduction~,
      tensor_publication~,
      shared_matrix_columns~,
      tensor_group?,
    ),
    text: GlmTextIo::new(
      vocabulary_shard?,
      head_law=if matrix_law == ShapeAwareFloat32 {
        ShapeAwareFloat32
      } else {
        ScalarFloat32
      },
      spec,
      maximum_rows~,
      maximum_output_rows~,
      maximum_workspace_bytes=maximum_text_workspace_bytes,
      boundary~,
    ),
    residual_bytes: text_plan.residual_bytes(),
    maximum_rows,
    maximum_history,
    vocabulary: spec.vocabulary_size(),
    maximum_sequences,
    maximum_outputs: maximum_output_rows,
    counts: None,
    activation_leased: false,
    active_rows: 0,
    submission_epoch: 0UL,
    residuals: [],
    status: Fresh,
  }
}

///|
pub fn GlmDecoderStage::layer_range(self : Self) -> (Int, Int) {
  self.range.layer_range()
}

///|
pub fn GlmDecoderStage::text_boundary(
  self : Self,
) -> @precision_ir.TextBoundary {
  self.text.boundary
}

///|
pub fn GlmDecoderStage::source_bytes(self : Self) -> Bytes {
  self.text.source_bytes() + b"\n" + self.decoder.source_bytes()
}

///|
/// Includes all owned weights, state, workspace and two boundary residuals.
/// Caller separately reserves borrowed module/stream/request ports and host IO.
pub fn GlmDecoderStage::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  self.decoder.required_device_bytes(checkpoint) +
  self.text.required_device_bytes(checkpoint) +
  2L * self.residual_bytes
}

///|
/// Budget the entire stage before its first allocation/upload. The text prefix
/// and suffix join the same decoder queue. Boundary residuals are never aliased.
pub fn GlmDecoderStage::prepare_inputs(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  ports : GlmDecoderInputs,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  collective? : @device.TensorCollectiveOwner,
  generation? : UInt64 = 1UL,
  retirement? : @parallel.RetirementPolicy = EachCollective,
  replay? : @parallel.ReplayPolicy = Eager,
  replay_budget_bytes? : Int64 = 0L,
  overlap_experts? : Bool = false,
  fuse_hyper? : Bool = false,
  fuse_controls? : Bool = false,
  defer_hyper? : Bool = false,
) -> Unit raise {
  guard self.status is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard ports.maximum_rows == self.maximum_rows &&
    ports.maximum_history <= self.maximum_history &&
    ports.vocabulary == self.vocabulary &&
    ports.maximum_sequences == self.maximum_sequences &&
    ports.maximum_outputs == self.maximum_outputs else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let decoder_bytes = self.decoder.required_device_bytes(checkpoint)
  let text_bytes = self.text.required_device_bytes(checkpoint)
  guard maximum_device_bytes >= 2L * self.residual_bytes &&
    text_bytes <= maximum_device_bytes - 2L * self.residual_bytes &&
    decoder_bytes <=
    maximum_device_bytes - 2L * self.residual_bytes - text_bytes else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  // Resolve all borrowed inputs before acquiring stage storage.
  let counts = ports.counts
  let ids = ports.token_ids
  let offsets = ports.sequence_offsets
  let slots = ports.sequence_slots
  let resets = ports.recurrent_resets
  let sparse_reset = ports.sparse_resets
  let output_count = ports.output_count
  let selected = ports.selected_rows
  let sampled = ports.sampled_tokens
  self.status = Failed
  self.residuals.push(context.allocate(self.residual_bytes))
  self.residuals.push(context.allocate(self.residual_bytes))
  self.text.prepare(
    context,
    aot,
    checkpoint,
    root,
    maximum_device_bytes=text_bytes,
    chunk_bytes~,
    limits~,
    counts~,
    token_ids=ids,
    output_count~,
    selected_rows=selected,
    input_residual=self.residuals[0],
    output_residual=self.residuals[1],
    sampled_tokens=sampled,
  )
  self.decoder.prepare(
    context,
    aot,
    stream,
    checkpoint,
    root,
    maximum_device_bytes=decoder_bytes,
    chunk_bytes~,
    limits~,
    counts~,
    sequence_offsets=offsets,
    sequence_slots=slots,
    reset_flags=resets,
    sparse_reset_flag=sparse_reset,
    input_positions?=ports.input_positions,
    page_ports?=ports.page_ports,
    hidden=self.residuals[0],
    output=self.residuals[1],
    prefix_launches=self.text.prefix(),
    suffix_launches=self.text.suffix(),
    vocabulary_boundary?=self.text.frame.vocabulary_boundary(),
    collective?,
    generation~,
    retirement~,
    replay~,
    replay_budget_bytes~,
    overlap_experts~,
    fuse_hyper~,
    fuse_controls~,
    defer_hyper~,
  )
  self.counts = Some(counts)
  self.status = Ready
}

///|
/// Borrow residual ports only for startup-bound pipeline transfer owners. The
/// producer must retire before the consumer reads and before storage release.
pub fn GlmDecoderStage::input_residual(self : Self) -> @device.Allocation raise {
  guard self.status is Ready else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.residuals[0]
}

///|
pub fn GlmDecoderStage::output_residual(
  self : Self,
) -> @device.Allocation raise {
  guard self.status is Ready else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.residuals[1]
}

///|
pub fn GlmDecoderStage::launch_count(self : Self) -> Int {
  self.decoder.launch_count()
}

///|
pub fn GlmDecoderStage::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] {
  self.decoder.append_descriptors()
}

///|
pub fn GlmDecoderStage::submit(self : Self, live_rows~ : Int) -> Unit raise {
  guard self.status is Ready && !self.activation_leased else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 &&
    live_rows <= self.maximum_rows &&
    self.submission_epoch < 0xffffffffffffffffUL else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.status = Failed
  self.decoder.submit(live_rows~)
  self.active_rows = live_rows
  self.submission_epoch += 1UL
  self.status = Active
}

///|
pub fn GlmDecoderStage::poll(self : Self) -> Bool raise {
  guard self.status is Active else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.status = Failed
  if self.decoder.poll() {
    self.status = Ready
    true
  } else {
    self.status = Active
    false
  }
}

///|
/// Rank error/sample retirement and activation DMA may both observe this one
/// submission. Once its queue has retired, neither observer polls it again.
fn GlmDecoderStage::poll_submission(self : Self) -> Bool raise {
  if self.status is Ready && self.active_rows > 0 {
    true
  } else {
    self.poll()
  }
}

///|
/// Stop execution without releasing allocations borrowed by worker DMA ports.
/// Captured collective callbacks must be dropped before communicator abort;
/// otherwise NCCL can wait for graphs that only later teardown would destroy.
pub fn GlmDecoderStage::abort_execution(self : Self) -> Unit raise {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.decoder.abort_execution()
}

///|
/// Close worker/transport leases first, then queue, text weights and residuals.
pub fn GlmDecoderStage::close(self : Self) -> Unit raise {
  if self.status is Closed {
    return
  }
  guard !self.activation_leased else {
    raise @device.DeviceError::Busy(OrderedExecutorClose)
  }
  self.status = Failed
  self.decoder.close()
  self.counts = None
  self.text.close()
  while self.residuals.last() is Some(buffer) {
    buffer.close()
    ignore(self.residuals.pop())
  }
  self.status = Closed
}
