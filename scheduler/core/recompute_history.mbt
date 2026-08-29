///|
fn Scheduler::history_stride(self : Scheduler) -> Int {
  self.capacity.maximum_generated_tokens_per_request()
}

///|
fn Scheduler::history_index(self : Scheduler, slot : Int, index : Int) -> Int {
  slot * self.history_stride() + index
}

///|
fn Scheduler::store_generated_token(
  self : Scheduler,
  slot : Int,
  output_index : Int,
  token : Int,
) -> Unit {
  if output_index < 0 || output_index >= self.history_stride() {
    abort("scheduler recompute history capacity invariant")
  }
  self.generated_history[self.history_index(slot, output_index)] = token
}

///|
fn Scheduler::recompute_target(self : Scheduler, slot : Int) -> Int {
  let request = self.requests[slot].unwrap()
  request.input_tokens().length() + self.output_counts[slot] - 1
}

///|
fn Scheduler::prefill_start(self : Scheduler, slot : Int) -> Int {
  if self.recompute_required[slot] {
    self.recompute_processed[slot]
  } else {
    self.prefill_cursors[slot]
  }
}

///|
fn Scheduler::prefill_target(self : Scheduler, slot : Int) -> Int {
  if self.recompute_required[slot] {
    self.recompute_target(slot)
  } else {
    self.requests[slot].unwrap().input_tokens().length()
  }
}

///|
fn Scheduler::prefill_token_at(
  self : Scheduler,
  slot : Int,
  index : Int,
) -> Int raise SchedulerError {
  let request = self.requests[slot].unwrap()
  let input_length = request.input_tokens().length()
  if index < input_length {
    request.input_token_at(index)
  } else {
    self.generated_history[self.history_index(slot, index - input_length)]
  }
}
