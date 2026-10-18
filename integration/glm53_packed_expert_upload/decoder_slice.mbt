///|
/// A contiguous checkpoint-backed decoder slice, suitable for one device or a
/// pipeline stage. Input/output are four-stream BF16 residual frames. This is
/// not an embedding, vocabulary head, sampler or distributed transport owner.
pub struct GlmDecoderSlice {
  priv first_layer : Int
  priv end_layer : Int
  priv model_layer_count : Int
  priv maximum_rows : Int
  priv residual_bytes : Int64
  priv buffer_count : Int
  priv layers : Array[GlmDecoderLayer]
  priv buffers : Array[@device.Allocation]
  priv descriptors : Array[@device.Allocation]
  priv mut queue : @device.OrderedKernelExecutor?
  priv tensor_group : (Int, Int)?
  priv mut collective_queue : @collective_program.CollectiveKernelProgram?
  priv mut collective_count : Int
  priv mut plan_sequence : UInt64
  priv mut global_sequence : UInt64
  priv mut launch_count : Int
  priv mut status : RecurrentLayerState
}

///|
/// Select the official hybrid layer schedule once. Per-layer ceilings constrain
/// pure plans; preparation additionally checks the aggregate device ceiling.
/// Sparse history currently represents one contiguous request, so a slice with
/// DSA layers cannot be prepared as multi-request attention.
pub fn GlmDecoderSlice::new(
  spec : @glm53_spec.Glm53FlashSpec,
  first_layer~ : Int,
  end_layer~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_history~ : Int,
  index_layer_norm_epsilon~ : Double,
  maximum_layer_state_bytes~ : Int64,
  maximum_layer_workspace_bytes~ : Int64,
  maximum_layer_expert_bytes~ : Int64,
  accuracy~ : @precision_ir.Accuracy,
  hc_reduction? : @precision_ir.HyperControlReduction = Ordered,
  recurrent_evaluation? : @precision_ir.RecurrentEvaluationLaw = OrderedTokenFloat32,
  matrix_law? : @precision_ir.DenseDotLaw = ScalarFloat32,
  matrix_staging? : @matrix.MatrixStaging = Synchronous,
  sparse_matrix_staging? : @matrix.MatrixStaging = Synchronous,
  expert_scaled_dot_law? : @precision_ir.ScaledDotReductionLaw = DecodeOperandsFloat32,
  expert_transport? : @matrix.ExpertTransport = IndexedRows,
  expert_fragment_slots? : Int = 1,
  expert_scale_partitions? : Int = 1,
  shared_matrix_columns? : Int = 64,
  tensor_group? : (Int, Int),
  paged_history? : @history_domain.PagedHistoryDomain,
) -> Self raise {
  guard first_layer >= 0 &&
    end_layer > first_layer &&
    end_layer <= spec.layer_count() &&
    maximum_rows > 0 else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let layers : Array[GlmDecoderLayer] = []
  if paged_history is Some(domain) {
    let (_, _, context, sequences) = domain.geometry()
    guard context == maximum_history && sequences == maximum_sequences else {
      raise @precision_ir.PrecisionPlanError::IllegalPlacement
    }
  }
  for layer_index in first_layer..<end_layer {
    let layer : GlmDecoderLayer = match spec.attention_kind(layer_index) {
      KimiDeltaAttention =>
        match spec.feed_forward_kind(layer_index) {
          DenseSwiGlu =>
            Recurrent(
              GlmRecurrentLayer::with_dense_block(
                spec,
                hc_reduction~,
                layer_index~,
                maximum_rows~,
                maximum_sequences~,
                cache_slots~,
                maximum_state_bytes=maximum_layer_state_bytes,
                evaluation=recurrent_evaluation,
                matrix_law~,
                matrix_staging~,
                tensor_group?,
              ),
            )
          RoutedMixtureOfExperts =>
            Recurrent(
              GlmRecurrentLayer::with_moe_block(
                spec,
                hc_reduction~,
                layer_index~,
                maximum_rows~,
                maximum_sequences~,
                cache_slots~,
                maximum_state_bytes=maximum_layer_state_bytes,
                maximum_expert_bytes=maximum_layer_expert_bytes,
                accuracy~,
                evaluation=recurrent_evaluation,
                matrix_law~,
                matrix_staging~,
                expert_scaled_dot_law~,
                expert_transport~,
                shared_matrix_columns~,
                expert_fragment_slots~,
                expert_scale_partitions~,
                tensor_group?,
              ),
            )
        }
      DeepSeekSparseAttention => {
        guard paged_history is Some(_) ||
          (maximum_sequences == 1 && cache_slots == 1) else {
          raise @precision_ir.PrecisionPlanError::IllegalPlacement
        }
        Sparse(
          GlmSparseLayer::new(
            spec,
            hc_reduction~,
            layer_index~,
            maximum_queries=maximum_rows,
            maximum_history~,
            paged_history?,
            index_layer_norm_epsilon~,
            maximum_state_bytes=maximum_layer_state_bytes,
            maximum_workspace_bytes=maximum_layer_workspace_bytes,
            maximum_expert_bytes=maximum_layer_expert_bytes,
            accuracy~,
            matrix_law~,
            matrix_staging=sparse_matrix_staging,
            expert_scaled_dot_law~,
            expert_transport~,
            shared_matrix_columns~,
            expert_fragment_slots~,
            expert_scale_partitions~,
            tensor_group?,
          ),
        )
      }
    }
    guard layer.workspace_bytes() <= maximum_layer_workspace_bytes &&
      layer.state_bytes() <= maximum_layer_state_bytes else {
      raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
    }
    layers.push(layer)
  }
  let count = end_layer - first_layer
  {
    first_layer,
    end_layer,
    model_layer_count: spec.layer_count(),
    maximum_rows,
    residual_bytes: maximum_rows.to_int64() *
    spec.hidden_size().to_int64() *
    spec.hyper_connection_streams().to_int64() *
    2L,
    buffer_count: (count - 1).min(2),
    layers,
    buffers: [],
    descriptors: [],
    queue: None,
    tensor_group,
    collective_queue: None,
    collective_count: 0,
    plan_sequence: 0UL,
    global_sequence: 1UL,
    launch_count: 0,
    status: Fresh,
  }
}

///|
pub fn GlmDecoderSlice::layer_range(self : Self) -> (Int, Int) {
  (self.first_layer, self.end_layer)
}

///|
pub fn GlmDecoderSlice::source_bytes(self : Self) -> Bytes {
  self.layers.fold(init=b"", (source, layer) => {
    source + b"\n" + layer.source_bytes()
  })
}

///|
/// Persistent state is included, as are at most two reusable residual buffers.
pub fn GlmDecoderSlice::workspace_bytes(self : Self) -> Int64 {
  self.layers.fold(init=self.residual_bytes * self.buffer_count.to_int64(), (
    bytes,
    layer,
  ) => bytes + layer.workspace_bytes())
}

///|
pub fn GlmDecoderSlice::required_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Int64 raise {
  let mut bytes = self.residual_bytes * self.buffer_count.to_int64()
  for layer in self.layers {
    bytes += layer.required_device_bytes(checkpoint)
  }
  bytes
}

///|
pub fn GlmDecoderSlice::device_bytes(self : Self) -> Int64 {
  self.layers.fold(
    init=self.residual_bytes * self.buffers.length().to_int64(),
    (bytes, layer) => bytes + layer.device_bytes(),
  )
}

///|
pub fn GlmDecoderSlice::launch_count(self : Self) -> Int {
  self.launch_count
}

///|
/// Bind all layer budgets before acquiring device memory. The two scratch
/// frames alternate; each layer consumes its predecessor's physical output.
pub fn GlmDecoderSlice::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  maximum_device_bytes~ : Int64,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  sparse_reset_flag~ : @device.Allocation,
  hidden~ : @device.Allocation,
  output~ : @device.Allocation,
  prefix_launches? : ArrayView[@device.OrderedKernelLaunch] = [],
  suffix_launches? : ArrayView[@device.OrderedKernelLaunch] = [],
  vocabulary_boundary? : (Int, @device.Allocation, @device.Allocation, Int, Int),
  collective? : @device.TensorCollectiveOwner,
  generation? : UInt64 = 1UL,
  retirement? : @parallel.RetirementPolicy = EachCollective,
  replay? : @parallel.ReplayPolicy = Eager,
  replay_budget_bytes? : Int64 = 0L,
  overlap_experts? : Bool = false,
  fuse_hyper? : Bool = false,
  fuse_controls? : Bool = false,
  defer_hyper? : Bool = false,
  input_positions? : @device.Allocation,
  page_ports? : @history_frame.SparseHistoryPagePorts,
) -> Unit raise {
  guard self.status is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard (self.tensor_group is Some(_)) == (collective is Some(_)) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  if self.tensor_group is Some((rank, world_size)) && collective is Some(owner) {
    owner.validate_group(rank~, world_size~)
  }
  let budgets = self.layers.map(layer => layer.required_device_bytes(checkpoint))
  let total = budgets.fold(
    init=self.residual_bytes * self.buffer_count.to_int64(),
    (bytes, layer_bytes) => bytes + layer_bytes,
  )
  guard total <= maximum_device_bytes else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  self.status = Failed
  for _ in 0..<self.buffer_count {
    self.buffers.push(context.allocate(self.residual_bytes))
  }
  let mut launches = prefix_launches.to_owned()
  let mut boundaries = []
  let mut forks = []
  let hyper_sites = []
  for index, layer in self.layers {
    let input = if index == 0 { hidden } else { self.buffers[(index - 1) % 2] }
    let next = if index == self.layers.length() - 1 {
      output
    } else {
      self.buffers[index % 2]
    }
    layer.prepare(
      context,
      aot,
      checkpoint,
      root,
      maximum_device_bytes=budgets[index],
      chunk_bytes~,
      limits~,
      counts~,
      sequence_offsets~,
      sequence_slots~,
      reset_flags~,
      sparse_reset_flag~,
      input_positions?,
      page_ports?,
      hidden=input,
      output=next,
    )
    for (sublayer, before_kernel, partials, width) in layer.tensor_boundaries() {
      boundaries.push(
        @collective_program.ReductionBoundary::new(
          before_kernel=launches.length() + before_kernel,
          width~,
          operation_id=(self.first_layer + index) * 3 + sublayer,
          buffer=partials,
        ),
      )
    }
    if overlap_experts && layer.producer_fork() is Some(span) {
      forks.push(
        span.shift(launches.length()) catch {
          _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
        },
      )
    }
    if fuse_hyper || fuse_controls || defer_hyper {
      for (first, last, frame) in layer.hyper_sites() {
        hyper_sites.push(
          (launches.length() + first, launches.length() + last, frame),
        )
      }
    }
    launches.append(layer.launches())
    if layer is Sparse(sparse) {
      self.descriptors.push(sparse.append_descriptor())
    }
  }
  if vocabulary_boundary is Some((cut, send, receive, width, world)) {
    boundaries.push(
      @collective_program.ReductionBoundary::fixed_gather(
        before_kernel=launches.length() + cut,
        local_width=width,
        world~,
        operation_id=self.end_layer * 3,
        send~,
        receive~,
      ),
    )
  }
  launches.append(suffix_launches)
  if fuse_hyper || fuse_controls || defer_hyper {
    let (fused, rewrite) = fuse_hyper_sites(
      launches,
      hyper_sites,
      aot,
      post_square=fuse_hyper,
      control_tail=fuse_controls,
      deferred=defer_hyper,
    )
    boundaries = boundaries.map(boundary => boundary.remap(rewrite))
    forks = forks.map(span => rewrite.fork(span))
    launches = fused
  }
  match collective {
    None =>
      self.queue = Some(
        context.create_ordered_kernel_executor(stream, launches),
      )
    Some(owner) => {
      self.collective_queue = Some(
        @collective_program.CollectiveKernelProgram::new(
          context,
          stream,
          launches,
          boundaries,
          owner~,
          generation~,
          maximum_rows=self.maximum_rows,
          retirement~,
          replay~,
          replay_budget_bytes~,
          forks~,
        ),
      )
      self.collective_count = boundaries.length()
    }
  }
  self.launch_count = launches.length()
  self.status = Ready
}

///|
/// Overflow ports, in sparse layer order, for model-level output delivery.
pub fn GlmDecoderSlice::append_descriptors(
  self : Self,
) -> ArrayView[@device.Allocation] {
  self.descriptors[:]
}
