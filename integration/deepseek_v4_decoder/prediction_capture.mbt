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
