///|
pub(all) struct Modules {
  decoder : @decoder.Modules
  text : @device.Module
}

///|
pub(all) struct Metadata {
  decoder : @decoder.Metadata
  token_ids : @device.Allocation
  output_count : @device.Allocation
  selected_rows : @device.Allocation
}

///|
/// Distinct embedding and output-head checkpoint tensors; never assume tying.
pub(all) struct Weights {
  embedding : @device.Allocation
  final_norm : @device.Allocation
  language_head : @device.Allocation
}

///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns the decoder, ingress/egress frames and combined phase queues. The reused
/// egress owner preallocates selected-row normalization and full-logit buffers.
/// All supplied modules/weights/metadata/state/activation/token buffers borrow.
pub struct Frame {
  priv program : Program
  priv decoder : @decoder.Frame
  priv ingress : @text_frame.TextIoFrame
  priv egress : @text_frame.TextIoFrame
  priv prefill : Array[@device.OrderedKernelLaunch]
  priv decode : Array[@device.OrderedKernelLaunch]
  priv mut status : FrameState
}

///|
pub fn Frame::new(program : Program) -> Self {
  {
    program,
    decoder: @decoder.Frame::new(program.decoder),
    ingress: @text_frame.TextIoFrame::new(program.text, boundary=Ingress),
    egress: @text_frame.TextIoFrame::new(program.text, boundary=Egress),
    prefill: [],
    decode: [],
    status: Fresh,
  }
}

///|
/// Caller supplies admitted token IDs, live counts, selected output rows and
/// output count. Close executors before this frame, then borrowed resources.
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  modules~ : Modules,
  metadata~ : Metadata,
  layers~ : ArrayView[@decoder.LayerWeights],
  weights~ : Weights,
  states~ : @decoder.StateArenas,
  buffers~ : @decoder.Buffers,
  sampled_tokens~ : @device.Allocation,
) -> Unit raise {
  guard self.status is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  self.decoder.prepare(
    context,
    modules=modules.decoder,
    metadata=metadata.decoder,
    layers~,
    states~,
    buffers~,
  )
  self.ingress.prepare_ingress(
    modules.text,
    counts=metadata.decoder.attention.counts,
    token_ids=metadata.token_ids,
    embedding=weights.embedding,
    input_residual=buffers.hidden,
  )
  self.egress.prepare_egress(
    context,
    modules.text,
    counts=metadata.decoder.attention.counts,
    output_count=metadata.output_count,
    selected_rows=metadata.selected_rows,
    output_residual=buffers.hidden,
    norm_weight=weights.final_norm,
    head_weight=weights.language_head,
    sampled_tokens~,
    maximum_workspace_bytes=self.program.workspace_bytes(),
  )
  for phase in [Phase::Prefill, Decode] {
    let queue = match phase {
      Prefill => self.prefill
      Decode => self.decode
    }
    for launch in self.ingress.prefix() {
      queue.push(launch)
    }
    for launch in self.decoder.launches(phase) {
      queue.push(launch)
    }
    for launch in self.egress.suffix() {
      queue.push(launch)
    }
  }
  self.status = Prepared
}

///|
pub fn Frame::launches(
  self : Self,
  phase : Phase,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  match phase {
    Prefill => self.prefill[:]
    Decode => self.decode[:]
  }
}

///|
/// Borrowed F32 storage of BF16-rounded logits, ordered by selected_rows.
pub fn Frame::logits(self : Self) -> @device.Allocation raise {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.egress.logits()
}

///|
pub fn Frame::close(self : Self) -> Unit raise {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.prefill.clear()
  self.decode.clear()
  self.egress.close()
  self.ingress.close()
  self.decoder.close()
  self.status = Closed
}
