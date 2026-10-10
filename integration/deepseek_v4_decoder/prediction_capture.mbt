///|
/// Local capture segments retain the official target-layer order. A stage
/// spanning only some targets owns only those segments; a distributed DSpark
/// consumer must assemble the entire ordered target set, never treat a partial
/// capture as complete prediction input.
pub fn DeepSeekDecoderSlice::prediction_capture_layers(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.capture_layers
}

///|
pub fn DeepSeekDecoderSlice::prediction_capture_bytes(self : Self) -> Int64 {
  match self.capture {
    Some(frame) => frame.plan().output_bytes()
    None => 0L
  }
}

///|
/// Borrowed until slice closure; visible only after preparation. Values become
/// ready when the same ordered decoder submission retires.
pub fn DeepSeekDecoderSlice::prediction_input(
  self : Self,
) -> @device.Allocation? raise {
  guard self.state is (Prepared | Active) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  match self.capture {
    Some(frame) => Some(frame.output())
    None => None
  }
}

///|
pub fn DeepSeekDecoderStage::prediction_capture_layers(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.decoder.prediction_capture_layers()
}

///|
pub fn DeepSeekDecoderStage::prediction_input(
  self : Self,
) -> @device.Allocation? raise {
  guard self.state is Prepared && !self.activation_leased else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.decoder.prediction_input()
}

///|
/// A local predictor may read captures only after base retirement. Hold this
/// lease through predictor completion so the next base step cannot overwrite
/// those captures or the shared vocabulary matrix during prediction.
pub fn DeepSeekDecoderStage::acquire_prediction_input(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared &&
    !self.activation_leased &&
    !self.prediction_leased &&
    self.prediction_complete &&
    self.active_rows > 0 &&
    self.decoder.prediction_input() is Some(input) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  self.prediction_leased = true
  input
}

///|
pub fn DeepSeekDecoderStage::release_prediction_input(
  self : Self,
) -> Unit raise {
  guard self.prediction_leased else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.prediction_leased = false
}

///|
pub fn DeepSeekDecoderStage::prediction_vocabulary(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared && self.egress is Some(egress) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  egress.vocabulary_weight()
}

///|
pub fn DeepSeekDecoderStage::prediction_counts(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared && self.counts is Some(counts) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  counts
}
