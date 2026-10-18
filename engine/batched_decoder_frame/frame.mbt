///|
pub(all) suberror BatchFrameError {
  InvalidCapacity
  Busy
  Closed
  Faulted
  CapacityExceeded
  InvalidToken
  InvalidPosition
  UnknownRequest
  GenerationMismatch
  ModelGenerationMismatch
} derive(Debug, Eq)

///|
pub extend BatchFrameError with @debug.Debug::{to_repr}

///|
pub extend BatchFrameError with Eq::{equal, not_equal}

///|
priv enum State {
  Idle
  Staged
  Failed
  Closed
}

///|
/// Batched ABI, deliberately distinct from the one-history serial decoder ABI.
/// Sequence ordinals are frame-local; slots persist until explicit retirement.
pub(all) enum BatchInputPort {
  Counts
  TokenIds
  SequenceOffsets
  SequenceSlots
  RecurrentResets
  SparseResets
  OutputCount
  SelectedRows
  Positions
  RowSlots
  SequencePositions
} derive(Debug, Eq)

///|
pub extend BatchInputPort with @debug.Debug::{to_repr}

///|
pub extend BatchInputPort with Eq::{equal, not_equal}

///|
fn BatchInputPort::index(self : Self) -> Int {
  match self {
    Counts => 0
    TokenIds => 1
    SequenceOffsets => 2
    SequenceSlots => 3
    RecurrentResets => 4
    SparseResets => 5
    OutputCount => 6
    SelectedRows => 7
    Positions => 8
    RowSlots => 9
    SequencePositions => 10
  }
}

///|
/// Preallocated effect owner for recurrent plus sparse-history decoders. It
/// does not choose schedules, sample tokens, allocate KV, or access the device.
/// The caller must bind a genuinely slot-addressed executor to these ports.
pub struct BatchedDecoderFrame {
  priv rows : Int
  priv sequences : Int
  priv slots : Int
  priv context : Int
  priv vocabulary : Int
  priv generation : UInt64
  priv inputs : FixedArray[FixedArray[Byte]]
  priv live : FixedArray[Bool]
  priv ids : FixedArray[UInt64]
  priv generations : FixedArray[UInt64]
  priv processed : FixedArray[Int]
  priv pending_slots : FixedArray[Int]
  priv pending_ids : FixedArray[UInt64]
  priv pending_generations : FixedArray[UInt64]
  priv pending_ends : FixedArray[Int]
  priv output_entries : FixedArray[Int]
  priv entry_outputs : FixedArray[Int]
  priv completion : @wire.FrameCompletionSlot
  priv mut active_sequences : Int
  priv mut active_rows : Int
  priv mut outputs : Int
  priv mut state : State
}

///|
pub fn BatchedDecoderFrame::new(
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_context~ : Int,
  vocabulary_size~ : Int,
  model_generation~ : UInt64,
) -> Self raise BatchFrameError {
  guard maximum_rows > 0 &&
    maximum_rows <= 536870910 &&
    maximum_sequences > 0 &&
    maximum_sequences <= maximum_rows &&
    cache_slots >= maximum_sequences &&
    cache_slots <= 536870910 &&
    maximum_context > 0 &&
    vocabulary_size > 0 else {
    raise InvalidCapacity
  }
  {
    rows: maximum_rows,
    sequences: maximum_sequences,
    slots: cache_slots,
    context: maximum_context,
    vocabulary: vocabulary_size,
    generation: model_generation,
    inputs: FixedArray::makei(11, port => {
      let bytes = match port {
        0 => 20
        6 => 4
        1 | 8 | 9 => maximum_rows * 4
        2 => (maximum_sequences + 1) * 4
        _ => maximum_sequences * 4
      }
      FixedArray::make(bytes, b'\x00')
    }),
    live: FixedArray::make(cache_slots, false),
    ids: FixedArray::make(cache_slots, 0UL),
    generations: FixedArray::make(cache_slots, 0UL),
    processed: FixedArray::make(cache_slots, 0),
    pending_slots: FixedArray::make(maximum_sequences, -1),
    pending_ids: FixedArray::make(maximum_sequences, 0UL),
    pending_generations: FixedArray::make(maximum_sequences, 0UL),
    pending_ends: FixedArray::make(maximum_sequences, 0),
    output_entries: FixedArray::make(maximum_sequences, -1),
    entry_outputs: FixedArray::make(maximum_sequences, -1),
    completion: @wire.FrameCompletionSlot::new(),
    active_sequences: 0,
    active_rows: 0,
    outputs: 0,
    state: Idle,
  }
}

///|
fn BatchedDecoderFrame::require_idle(self : Self) -> Unit raise BatchFrameError {
  match self.state {
    Idle => ()
    Staged => raise Busy
    Failed => raise Faulted
    Closed => raise Closed
  }
}

///|
fn BatchedDecoderFrame::require_staged(
  self : Self,
) -> Unit raise BatchFrameError {
  match self.state {
    Staged => ()
    Idle => raise Busy
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
pub fn BatchedDecoderFrame::input_capacity(
  self : Self,
  port : BatchInputPort,
) -> Int {
  self.inputs[port.index()].length()
}

///|
/// No new arrays are allocated during staging or transfer. Consumers use the
/// live counts; copying stale tail bytes is neither required nor permitted.
pub fn BatchedDecoderFrame::copy_input(
  self : Self,
  port : BatchInputPort,
  destination : FixedArray[Byte],
) -> Int raise BatchFrameError {
  self.require_staged()
  let bytes = match port {
    Counts => 20
    OutputCount => 4
    TokenIds | Positions | RowSlots => self.active_rows * 4
    SequenceOffsets => (self.active_sequences + 1) * 4
    SelectedRows => self.outputs * 4
    _ => self.active_sequences * 4
  }
  guard destination.length() >= bytes else { raise InvalidCapacity }
  let source = self.inputs[port.index()]
  for index in 0..<bytes {
    destination[index] = source[index]
  }
  bytes
}

///|
pub fn BatchedDecoderFrame::live_rows(self : Self) -> Int raise BatchFrameError {
  self.require_staged()
  self.active_rows
}

///|
pub fn BatchedDecoderFrame::output_rows(
  self : Self,
) -> Int raise BatchFrameError {
  self.require_staged()
  self.outputs
}

///|
/// Retains exact request-owned temperature/top-k/top-p/seed/sample-index values
/// through the original wire view. No greedy conversion or new RNG is inserted.
pub fn BatchedDecoderFrame::sampling_plan(
  self : Self,
) -> @wire.ValidatedPlanFrame raise {
  self.require_staged()
  self.completion.plan()
}

///|
/// Output index -> plan row, with prefill rows before decode rows. Non-final
/// prefill chunks have no output row. Use sampling_plan() with this mapping.
pub fn BatchedDecoderFrame::output_entry(
  self : Self,
  output : Int,
) -> Int raise BatchFrameError {
  self.require_staged()
  guard output >= 0 && output < self.outputs else { raise InvalidCapacity }
  self.output_entries[output]
}
