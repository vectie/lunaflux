///|
pub(all) suberror SerialDecoderError {
  InvalidCapacity
  Busy
  Closed
  Faulted
  UnsupportedBatch
  UnsupportedSampling
  InvalidToken
  InvalidPosition
  RequestStillLive
  ModelGenerationMismatch
} derive(Debug, Eq)

///|
pub extend SerialDecoderError with @debug.Debug::{to_repr}

///|
pub extend SerialDecoderError with Eq::{equal, not_equal}

///|
priv enum FrameState {
  Idle
  Staged
  Failed
  Closed
}

///|
/// Host-side request ports for a decoder whose history is one contiguous
/// sequence. This is independent of model family and device backend. Capacities
/// are fixed at startup; staging neither constructs a scheduler plan nor
/// allocates token arrays. GPU state advances only through complete().
pub struct SerialDecoderFrame {
  priv maximum_rows : Int
  priv maximum_context : Int
  priv vocabulary : Int
  priv model_generation : UInt64
  priv inputs : FixedArray[FixedArray[Byte]]
  priv mut state : FrameState
  priv mut request_live : Bool
  priv mut request_id : UInt64
  priv mut request_generation : UInt64
  priv mut processed : Int
  priv mut pending_id : UInt64
  priv mut pending_generation : UInt64
  priv mut pending_rows : Int
  priv mut pending_position : Int
  priv mut pending_output : Bool
  priv mut pending_prefill : Bool
  priv completion_slot : @wire.FrameCompletionSlot
}

///|
/// Input port order: counts, token IDs, sequence offsets, sequence slots,
/// recurrent resets, sparse reset, output count, selected rows. It is a physical
/// decoder-frame ABI, not a CUDA or GLM scheduling policy.
pub fn SerialDecoderFrame::new(
  maximum_rows~ : Int,
  maximum_context~ : Int,
  vocabulary_size~ : Int,
  model_generation~ : UInt64,
) -> Self raise SerialDecoderError {
  guard maximum_rows > 0 &&
    maximum_rows <= 536870911 &&
    maximum_context >= maximum_rows &&
    vocabulary_size > 0 else {
    raise InvalidCapacity
  }
  {
    maximum_rows,
    maximum_context,
    vocabulary: vocabulary_size,
    model_generation,
    inputs: [
      FixedArray::make(20, b'\x00'),
      FixedArray::make(maximum_rows * 4, b'\x00'),
      FixedArray::make(8, b'\x00'),
      FixedArray::make(4, b'\x00'),
      FixedArray::make(4, b'\x00'),
      FixedArray::make(4, b'\x00'),
      FixedArray::make(4, b'\x00'),
      FixedArray::make(4, b'\x00'),
    ],
    state: Idle,
    request_live: false,
    request_id: 0UL,
    request_generation: 0UL,
    processed: 0,
    pending_id: 0UL,
    pending_generation: 0UL,
    pending_rows: 0,
    pending_position: 0,
    pending_output: false,
    pending_prefill: false,
    completion_slot: @wire.FrameCompletionSlot::new(),
  }
}

///|
/// Startup compatibility facts; inspecting them does not stage a request.
pub fn SerialDecoderFrame::geometry(self : Self) -> (Int, Int, Int) {
  (self.maximum_rows, self.maximum_context, self.vocabulary)
}

///|
fn SerialDecoderFrame::require_idle(
  self : Self,
) -> Unit raise SerialDecoderError {
  match self.state {
    Idle => ()
    Staged => raise Busy
    Failed => raise Faulted
    Closed => raise Closed
  }
}

///|
fn store_i32(buffer : FixedArray[Byte], index : Int, value : Int) -> Unit {
  let bits = value.reinterpret_as_uint()
  for byte in 0..<4 {
    buffer[index * 4 + byte] = ((bits >> (byte * 8)) & 255U).to_byte()
  }
}

///|
/// Validate all input semantics before touching the prepared payload. A new
/// request starts at position zero. Continuation must match both request
/// generation and the exact committed context length; no prefix state is
/// invented for an unseen request. release_request() explicitly permits reuse.
pub fn SerialDecoderFrame::stage(
  self : Self,
  frame : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  self.require_idle()
  writer.validate_plan(frame)
  guard frame.model_generation_value() == self.model_generation else {
    raise SerialDecoderError::ModelGenerationMismatch
  }
  let prefills = frame.prefill_count()
  let decodes = frame.decode_count()
  guard prefills + decodes == 1 else {
    raise SerialDecoderError::UnsupportedBatch
  }
  let (id, generation, position, start, rows, output, sampling) = if prefills ==
    1 {
    let row = frame.prefill_row(0)
    (
      row.request_id_value(),
      row.request_generation_value(),
      row.sequence_token_start(),
      row.token_start(),
      row.token_length(),
      row.produces_token(),
      row.sampling_mode_value(),
    )
  } else {
    let row = frame.decode_row(0)
    (
      row.request_id_value(),
      row.request_generation_value(),
      row.sequence_token_position(),
      row.token_start(),
      1,
      true,
      row.sampling_mode_value(),
    )
  }
  guard !output || sampling == 0 else {
    raise SerialDecoderError::UnsupportedSampling
  }
  guard rows > 0 &&
    rows <= self.maximum_rows &&
    start == 0 &&
    rows == frame.token_count() &&
    position >= 0 &&
    position <= self.maximum_context - rows else {
    raise SerialDecoderError::InvalidPosition
  }
  if self.request_live {
    guard id == self.request_id && generation == self.request_generation else {
      raise SerialDecoderError::RequestStillLive
    }
    guard position == self.processed else {
      raise SerialDecoderError::InvalidPosition
    }
  } else {
    guard prefills == 1 && position == 0 else {
      raise SerialDecoderError::InvalidPosition
    }
  }
  for index in 0..<rows {
    let token = frame.token_at(index)
    guard token >= 0 && token < self.vocabulary else {
      raise SerialDecoderError::InvalidToken
    }
  }
  for index in 0..<rows {
    store_i32(self.inputs[1], index, frame.token_at(index))
  }
  store_i32(self.inputs[0], 0, prefills)
  store_i32(self.inputs[0], 1, decodes)
  store_i32(self.inputs[0], 2, 1)
  store_i32(self.inputs[0], 3, rows)
  store_i32(self.inputs[2], 0, 0)
  store_i32(self.inputs[2], 1, rows)
  store_i32(self.inputs[3], 0, 0)
  let reset = if self.request_live { 0 } else { 1 }
  store_i32(self.inputs[4], 0, reset)
  store_i32(self.inputs[5], 0, reset)
  store_i32(self.inputs[6], 0, if output { 1 } else { 0 })
  store_i32(self.inputs[7], 0, rows - 1)
  self.pending_id = id
  self.pending_generation = generation
  self.pending_rows = rows
  self.pending_position = position
  self.pending_prefill = prefills == 1
  self.pending_output = output
  self.completion_slot.bind(frame, writer)
  self.state = Staged
}

///|
/// Copy the staged payload into a prepared transfer's pinned storage. Only
/// the live token prefix is copied; the immutable envelope lengths are fixed.
pub fn SerialDecoderFrame::copy_input(
  self : Self,
  port : Int,
  destination : FixedArray[Byte],
) -> Int raise SerialDecoderError {
  guard self.state is Staged else { raise Busy }
  guard port >= 0 && port < 8 else { raise InvalidCapacity }
  let source = self.inputs[port]
  let length = if port == 1 { self.pending_rows * 4 } else { source.length() }
  guard destination.length() >= length else { raise InvalidCapacity }
  for index in 0..<length {
    destination[index] = source[index]
  }
  length
}

///|
pub fn SerialDecoderFrame::input_capacity(
  self : Self,
  port : Int,
) -> Int raise SerialDecoderError {
  guard port >= 0 && port < 8 else { raise InvalidCapacity }
  self.inputs[port].length()
}

///|
pub fn SerialDecoderFrame::live_rows(
  self : Self,
) -> Int raise SerialDecoderError {
  guard self.state is Staged else { raise Busy }
  self.pending_rows
}

///|
pub fn SerialDecoderFrame::produces_token(
  self : Self,
) -> Bool raise SerialDecoderError {
  guard self.state is Staged else { raise Busy }
  self.pending_output
}

///|
/// Publish only after device completion and all device error ports have been
/// checked. Invalid samples become execution failures, not successful tokens.
/// Any executed failure terminally poisons the retained history owner.
pub fn SerialDecoderFrame::complete(
  self : Self,
  sampled_token~ : Int,
  execution_failed~ : Bool,
) -> @wire.ValidatedCompletionFrame raise {
  guard self.state is Staged else { raise SerialDecoderError::Busy }
  let frame = self.completion_slot.plan()
  let writer = self.completion_slot.writer()
  self.state = Failed
  let failed = execution_failed ||
    (
      self.pending_output &&
      (sampled_token < 0 || sampled_token >= self.vocabulary)
    )
  if self.pending_prefill {
    let row = frame.prefill_row(0)
    if failed {
      writer.append_prefill_failed(row, ExecutionFailed)
    } else if self.pending_output {
      writer.append_final_prefill_completed(row, sampled_token)
    } else {
      writer.append_prefill_completed(row)
    }
  } else {
    let row = frame.decode_row(0)
    if failed {
      writer.append_decode_failed(row, ExecutionFailed)
    } else {
      writer.append_decode_completed(row, sampled_token)
    }
  }
  let completed = self.completion_slot.submit()
  if !failed {
    self.request_live = true
    self.request_id = self.pending_id
    self.request_generation = self.pending_generation
    self.processed = self.pending_position + self.pending_rows
    self.state = Idle
  }
  completed
}

///|
/// Retire a partial decoder stage only after the pipeline coordinator accepts
/// the whole step. No model completion or token is published by this rank.
/// A local or peer execution failure poisons retained history; it cannot be
/// rolled back by retrying the same position. The local writer lease is aborted.
pub fn SerialDecoderFrame::retire_stage(
  self : Self,
  execution_failed~ : Bool,
) -> Unit raise {
  guard self.state is Staged else { raise SerialDecoderError::Busy }
  self.state = Failed
  self.completion_slot.close()
  if !execution_failed {
    self.request_live = true
    self.request_id = self.pending_id
    self.request_generation = self.pending_generation
    self.processed = self.pending_position + self.pending_rows
    self.state = Idle
  }
}

///|
/// Call only after the request is retired by its scheduler. The next prefill
/// explicitly resets both recurrent and sparse history before processing.
pub fn SerialDecoderFrame::release_request(
  self : Self,
) -> Unit raise SerialDecoderError {
  self.require_idle()
  self.request_live = false
  self.processed = 0
}

///|
pub fn SerialDecoderFrame::processed_tokens(self : Self) -> Int {
  self.processed
}

///|
pub fn SerialDecoderFrame::is_faulted(self : Self) -> Bool {
  self.state is Failed
}

///|
/// Read the last published completion until its wire owner is reused.
pub fn SerialDecoderFrame::completion(
  self : Self,
) -> @wire.ValidatedCompletionFrame raise {
  guard !(self.state is Closed) else { raise SerialDecoderError::Closed }
  self.completion_slot.completion()
}

///|
/// Discard the open completion when cancellation or device execution fails.
/// The caller drains its GPU executor before closing borrowed buffers.
pub fn SerialDecoderFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.completion_slot.close()
  self.state = Closed
}
