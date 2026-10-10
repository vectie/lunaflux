///|
/// Model binding for the common plaintext TCP/pinned-DMA activation link.
/// Owns the link, borrows the prepared stage, and protects its residual storage
/// until transfer retirement. Close before the stage or its request ports.
pub struct DeepSeekActivationEdge {
  priv stage : DeepSeekDecoderStage
  priv link : @activation.ActivationLink
  priv outgoing : Bool
  priv mut prepared : Bool
  priv mut leased : Bool
  priv mut last_submission : UInt64
}

///|
pub fn DeepSeekActivationEdge::new(
  stage : DeepSeekDecoderStage,
  connection : @socket.Tcp,
  outgoing~ : Bool,
  maximum_host_bytes~ : Int64,
  timeout_millis~ : UInt64,
) -> Self raise {
  guard stage.state is Prepared && !stage.activation_leased else {
    raise @activation.ActivationLinkError::InvalidState
  }
  let boundary = stage.text_boundary()
  if (outgoing && !(boundary is (Ingress | Interior))) ||
    (!outgoing && !(boundary is (Egress | Interior))) {
    raise @activation.ActivationLinkError::InvalidState
  }
  let row_bytes = (stage.residual_bytes / stage.maximum_rows.to_int64()).to_int()
  let link = if outgoing {
    @activation.ActivationLink::sender(
      connection,
      maximum_rows=stage.maximum_rows,
      row_bytes~,
      maximum_host_bytes~,
      timeout_millis~,
      producer_ready=() => stage.poll_submission(),
    )
  } else {
    @activation.ActivationLink::receiver(
      connection,
      maximum_rows=stage.maximum_rows,
      row_bytes~,
      maximum_host_bytes~,
      timeout_millis~,
    )
  }
  {
    stage,
    link,
    outgoing,
    prepared: false,
    leased: false,
    last_submission: 0UL,
  }
}

///|
pub fn DeepSeekActivationEdge::prepare(
  self : Self,
  context : @device.Context,
) -> Unit raise {
  guard !self.prepared && self.stage.state is Prepared else {
    raise @activation.ActivationLinkError::InvalidState
  }
  let allocation = if self.outgoing {
    self.stage.output_residual()
  } else {
    self.stage.input_residual()
  }
  self.link.prepare(context, allocation)
  self.prepared = true
}

///|
pub fn DeepSeekActivationEdge::begin(
  self : Self,
  epoch~ : UInt64,
  live_rows~ : Int,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard self.prepared &&
    !self.leased &&
    !self.stage.activation_leased &&
    (if self.outgoing {
      self.stage.state is (Active | Prepared) &&
      self.stage.submission_epoch > self.last_submission &&
      self.stage.active_rows > 0 &&
      live_rows == self.stage.active_rows
    } else {
      self.stage.state is Prepared
    }) else {
    raise @activation.ActivationLinkError::InvalidState
  }
  self.link.begin(epoch~, live_rows~, local_now_millis~)
  if self.outgoing {
    self.last_submission = self.stage.submission_epoch
  }
  self.stage.activation_leased = true
  self.leased = true
}

///|
pub async fn DeepSeekActivationEdge::poll(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard self.leased else { raise @activation.ActivationLinkError::InvalidState }
  if self.link.poll(local_now_millis~) {
    self.stage.activation_leased = false
    self.leased = false
    true
  } else {
    false
  }
}

///|
/// The common link drains pending DMA before returning the residual lease.
pub fn DeepSeekActivationEdge::close(self : Self) -> Unit raise {
  self.link.close()
  if self.leased {
    self.stage.activation_leased = false
    self.leased = false
  }
  self.prepared = false
}
