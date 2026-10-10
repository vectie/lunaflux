///|
priv enum State {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Executable local predictor attached to the base egress stage. The base stage
/// owns its captures and vocabulary; this owner streams a local embedding copy
/// (the base embedding may live on a different rank). Its bytes are not hidden
/// in the base placement reserve. Verification/commit remains a separate effect.
pub struct DeepSeekPredictionExecution {
  priv spec : @spec.DeepSeekV4ModelSpec
  priv main_rows : Int
  priv output_rows : Int
  priv explicit_seed : Bool
  priv mut seed_input : @text_frame.CommittedTokenSeed?
  priv mut seed_pending : Bool
  priv predictor : @prediction.DeepSeekPredictor
  priv embedding : @upload.CheckpointDeviceWeights
  priv results : Array[@device.Allocation]
  priv mut base : @decoder.DeepSeekDecoderStage?
  priv mut queues : @execution.PredictionPhaseExecution?
  priv mut leased : Bool
  priv mut ready_result : Bool
  priv mut state : State
}

///|
pub fn DeepSeekPredictionExecution::new(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_main_rows~ : Int,
  maximum_positions~ : Int,
  maximum_workspace_bytes~ : Int64,
  maximum_expert_bytes~ : Int64,
  temperature~ : Double,
  accuracy~ : @precision.Accuracy,
  maximum_output_rows? : Int = 1,
  explicit_committed_seed? : Bool = false,
) -> Self raise {
  {
    spec,
    main_rows: maximum_main_rows,
    output_rows: maximum_output_rows,
    explicit_seed: explicit_committed_seed,
    seed_input: None,
    seed_pending: false,
    predictor: @prediction.DeepSeekPredictor::new(
      spec,
      maximum_main_rows~,
      maximum_positions~,
      maximum_workspace_bytes~,
      maximum_expert_bytes~,
      temperature~,
      accuracy~,
      maximum_output_rows=if explicit_committed_seed {
        1
      } else {
        maximum_output_rows
      },
    ),
    embedding: @upload.CheckpointDeviceWeights::new(),
    results: [],
    base: None,
    queues: None,
    leased: false,
    ready_result: false,
    state: Fresh,
  }
}

///|
pub fn DeepSeekPredictionExecution::source_bytes(self : Self) -> Bytes raise {
  self.predictor.source_bytes()
}

///|
/// Additional bytes beside the already budgeted base stage and request ports.
pub fn DeepSeekPredictionExecution::required_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Int64 raise {
  self.predictor.required_device_bytes(checkpoint) +
  @text.bind_text_ingress_weight(checkpoint, self.spec).buffer_bytes() +
  (self.spec.dspark_block_size().to_int64() + 1L) * 4L +
  self.spec.dspark_block_size().to_int64() * 4L +
  16L +
  (if self.explicit_seed { 4L } else { 0L })
}

///|
pub fn DeepSeekPredictionExecution::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  base : @decoder.DeepSeekDecoderStage,
  ports : @serial.SerialDecoderWorker,
  rng : Bytes,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming.StreamingSafetensorsLimits,
) -> Unit raise {
  guard self.state is Fresh &&
    rng.length() == 16 &&
    chunk_bytes > 0 &&
    base.prediction_capture_layers()[:] ==
    self.spec.dspark_target_layer_ids()[:] &&
    base.prediction_input() is Some(captured) &&
    base.prediction_counts().same_storage(ports.input(Counts)) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (rows, _, vocabulary_size) = ports.geometry()
  guard rows == self.main_rows &&
    vocabulary_size == self.spec.vocabulary_size() &&
    ports.output_capacity() == self.output_rows else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard self.required_device_bytes(checkpoint) <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  // A last-stage vocabulary is shared, never silently borrowed across contexts.
  let vocabulary = base.prediction_vocabulary()
  context.validate_allocation_region(
    vocabulary,
    offset=0L,
    byte_count=self.spec.vocabulary_size().to_int64() *
      self.spec.hidden_size().to_int64() *
      2L,
    alignment=2L,
  )
  self.state = Failed
  self.base = Some(base)
  let weight = @text.bind_text_ingress_weight(checkpoint, self.spec)
  self.embedding.upload(
    context,
    checkpoint,
    root,
    [weight],
    maximum_device_bytes=weight.buffer_bytes(),
    chunk_bytes~,
    limits~,
  )
  for
    bytes in [
      (self.spec.dspark_block_size().to_int64() + 1L) * 4L,
      self.spec.dspark_block_size().to_int64() * 4L,
      16L,
    ] {
    self.results.push(context.allocate(bytes))
  }
  self.results[2].copy_from_host(rng, 0L, 0L, 16L)
  if self.explicit_seed {
    self.seed_input = Some(
      @text_frame.CommittedTokenSeed::new(
        context,
        vocabulary_size=self.spec.vocabulary_size(),
      ),
    )
  }
  self.predictor.prepare(
    context,
    aot,
    checkpoint,
    root,
    main_counts=ports.input(Counts),
    main_positions=ports.input(Positions),
    captured_hidden=captured,
    target_layers=base.prediction_capture_layers(),
    embedding_weight=self.embedding.allocation(0),
    vocabulary_weight=vocabulary,
    seed_token=match self.seed_input {
      Some(seed) => seed.allocation()
      None => ports.sampled_tokens()
    },
    reset_flag=ports.input(SparseReset),
    start_position=ports.input(Positions),
    tokens=self.results[0],
    confidence=self.results[1],
    rng=self.results[2],
    maximum_device_bytes=self.predictor.required_device_bytes(checkpoint),
    chunk_bytes~,
    limits~,
  )
  self.queues = Some(
    @execution.PredictionPhaseExecution::prepare(
      context,
      stream,
      self.predictor.prime_launches(),
      self.predictor.prediction_launches(),
      committed=self.predictor.prediction_from_committed_launches(),
    ),
  )
  self.state = Prepared
}

///|
/// Run after the corresponding committed base submission retires. Prime is for
/// prompt/accepted chunks; Predict appends one main row, while PredictCommitted
/// consumes an already-primed prefix without publishing the same positions twice.
pub fn DeepSeekPredictionExecution::submit(
  self : Self,
  phase : @execution.PredictionPhase,
) -> Unit raise {
  guard self.state is Prepared &&
    !self.leased &&
    self.base is Some(base) &&
    self.queues is Some(queues) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard !self.explicit_seed || phase is Prime else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  ignore(base.acquire_prediction_input())
  self.leased = true
  self.ready_result = false
  queues.submit(phase) catch {
    e => {
      queues.abort()
      base.release_prediction_input()
      self.leased = false
      raise e
    }
  }
  self.ready_result = !(phase is Prime)
}

///|
pub fn DeepSeekPredictionExecution::poll(self : Self) -> Bool raise {
  guard self.state is Prepared &&
    self.leased &&
    self.base is Some(base) &&
    self.queues is Some(queues) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  if self.seed_pending {
    guard self.seed_input is Some(seed) else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
    }
    if !seed.poll() {
      return false
    }
    self.seed_pending = false
    queues.submit(PredictCommitted) catch {
      error => {
        queues.abort()
        base.release_prediction_input()
        self.leased = false
        raise error
      }
    }
  }
  if !queues.poll() {
    return false
  }
  base.release_prediction_input()
  self.leased = false
  true
}

///|
/// Startup-only binding for a prepared result readback. This does not make the
/// uninitialized/pending tokens readable; submit/poll still owns execution.
/// The readback must retire and close before this owner releases the output.
pub fn DeepSeekPredictionExecution::borrow_draft_output(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared && !self.leased else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.results[0]
}

///|
/// [base seed, five dependent samples], only after prediction retirement.
pub fn DeepSeekPredictionExecution::tokens(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared && !self.leased && self.ready_result else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.results[0]
}

///|
pub fn DeepSeekPredictionExecution::confidence(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared && !self.leased && self.ready_result else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.results[1]
}

///|
/// Close before the borrowed base stage. Drain queues before releasing the
/// capture lease and private allocations, including partial-startup failures.
pub fn DeepSeekPredictionExecution::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.ready_result = false
  if self.queues is Some(queues) {
    queues.close()
    self.queues = None
  }
  if self.leased && self.base is Some(base) {
    base.release_prediction_input()
    self.leased = false
  }
  self.predictor.close()
  if self.seed_input is Some(seed) {
    seed.close()
    self.seed_input = None
  }
  while self.results.last() is Some(buffer) {
    buffer.close()
    ignore(self.results.pop())
  }
  self.embedding.close()
  self.base = None
  self.state = Closed
}
