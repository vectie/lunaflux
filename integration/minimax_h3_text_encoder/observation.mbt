///|
/// Pure partition derived from the same layer recipe as binding. Include the
/// visual embedding replacement and three post-layer deepstack additions.
fn hidden_state_boundaries(visual : Bool) -> ReadOnlyArray[(Int, Int)] {
  let embedding_end = if visual { 2 } else { 1 }
  let ends : Array[(Int, Int)] = [(0, embedding_end)]
  for layer in 0..<50; end = embedding_end {
    let visual_steps = if visual && deepstack_stage(layer) is Some(_) {
      1
    } else {
      0
    }
    let next = end + layer_function_order().length() + visual_steps
    ends.push((layer + 1, next))
    continue next
  }
  ReadOnlyArray::from_array(ends)
}

///|
/// Explicit numerical diagnostic, not a serving/performance execution path.
/// Synchronizes and copies the current hidden state after embedding and each
/// full decoder layer. Failed observation cannot mark the request complete or
/// rerun it; close/cancel deterministically release the prepared queue owners.
pub fn EncoderRequest::execute_observed(
  self : Self,
  on_hidden_state~ : (Int, Bytes) -> Unit raise,
  original_hidden_state? : (Int) -> Bytes raise,
) -> Unit raise {
  guard self.phase == Prepared &&
    self.queue is None &&
    self.observations.length() == 51 &&
    !self.allocations.is_empty() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  observe_hidden_states(
    self.observations.length(),
    index => {
      // Offline attribution only: replace the previous layer's input with its
      // literal original state. Never propagate a diagnostic output as the
      // next matched input, or change any kernel/rounding law.
      if matched_layer_input(
          index,
          self.source.plan.output_bytes(),
          original_hidden_state,
        )
        is Some(bytes) {
        self.allocations[0].copy_from_host(
          bytes,
          0L,
          0L,
          self.source.plan.output_bytes(),
        )
      }
      let (queue, _) = self.observations[index]
      for index in 0..<queue.kernel_count() {
        queue.enqueue(index)
      }
      queue.record_completion()
      queue.wait_completion()
      self.allocations[0].copy_to_host(0L, self.source.plan.output_bytes())
    },
    (index, snapshot) => on_hidden_state(self.observations[index].1, snapshot),
  )
  self.phase = Complete
}

///|
fn matched_layer_input(
  next_hidden : Int,
  output_bytes : Int64,
  original : ((Int) -> Bytes raise)?,
) -> Bytes? raise {
  if next_hidden == 0 || original is None {
    return None
  }
  let bytes = original.unwrap()(next_hidden - 1)
  guard bytes.length().to_int64() == output_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  Some(bytes)
}

///|
fn observe_hidden_states(
  count : Int,
  run : (Int) -> Bytes raise,
  consume : (Int, Bytes) -> Unit raise,
) -> Unit raise {
  for hidden in 0..<count {
    let snapshot = run(hidden)
    consume(hidden, snapshot)
  }
}
