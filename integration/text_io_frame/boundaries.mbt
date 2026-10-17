///|
fn TextIoFrame::bind_ingress(
  self : Self,
  aot : @device.Module,
  counts : @device.KernelArgument,
  ids : @device.KernelArgument,
  embedding : @device.KernelArgument,
  residual : @device.KernelArgument,
) -> Unit raise {
  self.before.push(
    self.launch(
      aot,
      0,
      (self.source.plan().residual_bytes() / 2L + 255L) / 256L,
      256,
      [counts, ids, embedding, residual],
    ),
  )
}

///|
fn TextIoFrame::bind_egress(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts : @device.KernelArgument,
  output_count : @device.KernelArgument,
  selected : @device.KernelArgument,
  residual : @device.KernelArgument,
  norm : @device.KernelArgument,
  head : @device.KernelArgument,
  tokens : @device.KernelArgument,
) -> Unit raise {
  let plan = self.source.plan()
  let (_, _, _, _, outputs) = plan.geometry()
  self.buffers.push(context.allocate(plan.normalized_bytes()))
  self.buffers.push(context.allocate(self.source.local_logits_bytes()))
  let normalized = region(self.buffers[0], plan.normalized_bytes(), 2L)
  let logits = region(self.buffers[1], self.source.local_logits_bytes(), 4L)
  let candidates = if self.source.vocabulary_shard() is Some(shard) {
    let bytes = shard.candidate_width().to_int64() * 4L
    self.buffers.push(context.allocate(bytes))
    self.buffers.push(context.allocate(bytes * shard.group().1.to_int64()))
    region(self.buffers[2], bytes, 4L)
  } else {
    tokens
  }
  for stage in self.source.egress_order() {
    let launch = match stage {
      0 =>
        self.launch(aot, 1, outputs.to_int64(), 256, [
          counts, output_count, selected, residual, norm, normalized,
        ])
      1 =>
        self.launch(
          aot,
          2,
          self.source.head_launch().0.to_int64(),
          self.source.head_launch().1,
          [output_count, normalized, head, logits],
        )
      2 =>
        self.launch(aot, 3, outputs.to_int64(), self.source.greedy_threads(), [
          counts, output_count, selected, logits, candidates,
        ])
      _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    self.after.push(launch)
  }
  if self.source.vocabulary_shard() is Some(shard) {
    self.after.push(
      self.launch(aot, 4, outputs.to_int64(), 1, [
        output_count,
        region(
          self.buffers[3],
          shard.candidate_width().to_int64() * shard.group().1.to_int64() * 4L,
          4L,
        ),
        tokens,
      ]),
    )
  }
}

///|
/// First-stage binding requires no output weights, output ports or workspace.
pub fn TextIoFrame::prepare_ingress(
  self : Self,
  aot : @device.Module,
  counts~ : @device.Allocation,
  token_ids~ : @device.Allocation,
  embedding~ : @device.Allocation,
  input_residual~ : @device.Allocation,
) -> Unit raise {
  guard self.state is Fresh && self.boundary is Ingress else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (_, _, _, rows, _) = plan.geometry()
  let counts = region(counts, 20L, 4L)
  let ids = region(token_ids, rows.to_int64() * 4L, 4L)
  let embedding = region(embedding, plan.table_bytes(), 2L)
  let residual = region(input_residual, plan.residual_bytes(), 2L)
  self.bind_ingress(aot, counts, ids, embedding, residual)
  self.state = Prepared
}

///|
/// Last-stage binding requires no token-ID or embedding allocation.
pub fn TextIoFrame::prepare_egress(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  output_count~ : @device.Allocation,
  selected_rows~ : @device.Allocation,
  output_residual~ : @device.Allocation,
  norm_weight~ : @device.Allocation,
  head_weight~ : @device.Allocation,
  sampled_tokens~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    self.boundary is Egress &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  self.replay_inputs.append([counts, output_count, selected_rows])
  let plan = self.source.plan()
  let (_, hidden, _, _, outputs) = plan.geometry()
  let counts = region(counts, 20L, 4L)
  let count = region(output_count, 4L, 4L)
  let selected = region(selected_rows, outputs.to_int64() * 4L, 4L)
  let residual = region(output_residual, plan.residual_bytes(), 2L)
  let norm = region(norm_weight, hidden.to_int64() * 2L, 2L)
  let head = region(head_weight, self.source.head_bytes(), 2L)
  let tokens = region(sampled_tokens, outputs.to_int64() * 4L, 4L)
  self.bind_egress(
    context, aot, counts, count, selected, residual, norm, head, tokens,
  )
  self.state = Prepared
}

///|
/// Interior stages carry residuals without any text kernels or global weights.
pub fn TextIoFrame::prepare_interior(self : Self) -> Unit raise {
  guard self.state is Fresh && self.boundary is Interior else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Prepared
}
