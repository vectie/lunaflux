///|
/// Immutable startup-only prompt replay. Capacity and the contiguous request
/// layout are resolved before device preparation. Execution consumes these same
/// byte snapshots, never reopens the prompt files or authenticates their payloads.
pub struct PreparedSerialPrompt {
  priv frames : ReadOnlyArray[Bytes]
  priv input_tokens : Int
  priv maximum_new_tokens : Int
  priv encoded_bytes : Int64
}

///|
pub fn PreparedSerialPrompt::new(
  frames : ArrayView[Bytes],
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  maximum_new_tokens~ : Int,
  maximum_context~ : Int,
  maximum_encoded_bytes~ : Int64,
  model_generation~ : UInt64,
) -> Self raise {
  guard maximum_new_tokens > 0 &&
    maximum_new_tokens <= inference.max_new_tokens() &&
    maximum_context > maximum_new_tokens &&
    maximum_context <= limits.max_sequence_tokens() &&
    maximum_context <= inference.max_context_tokens() &&
    !frames.is_empty() &&
    frames.length() <= maximum_context - maximum_new_tokens &&
    maximum_encoded_bytes > 0L &&
    model_generation > 0UL else {
    fail("invalid bounded serial prompt replay")
  }
  let owner = @wire.PlanFrameBuffer::new(limits, inference)
  let scratch = FixedArray::make(owner.capacity(), b'\x00')
  // Functional state holds only scalar geometry/identity. No CUDA, model-family
  // branching, hashing or filesystem work belongs in this transformation.
  // Invariants: end <= context - generation; used <= encoded-byte budget.
  // Validated rows are nonempty; each iteration advances the prompt end.
  // Compare before addition so neither the context nor byte sum can overflow.
  let (input_tokens, encoded_bytes) = for
    index, bytes in frames
    end = 0, used = 0L, request_id = 0UL, request_generation = 0UL {
    guard bytes.length() <= owner.capacity() &&
      bytes.length().to_int64() <= maximum_encoded_bytes - used else {
      fail("prompt replay exceeds its retained byte budget")
    }
    for offset, byte in bytes {
      scratch[offset] = byte
    }
    let frame = owner.load(scratch, bytes.length())
    guard frame.prefill_count() == 1 &&
      frame.decode_count() == 0 &&
      frame.sequence_value() == index.to_uint64() + 1UL &&
      frame.model_generation_value() == model_generation else {
      fail("prompt replay requires one ordered prefill request")
    }
    let row = frame.prefill_row(0)
    let id = row.request_id_value()
    let generation = row.request_generation_value()
    let length = row.token_length()
    guard row.sequence_token_start() == end &&
      row.produces_token() == (index == frames.length() - 1) &&
      length <= maximum_context - maximum_new_tokens - end &&
      (index == 0 || (id == request_id && generation == request_generation)) else {
      fail(
        "prompt and generated decode positions exceed the reserved context or are not contiguous",
      )
    }
    continue end + length, used + bytes.length().to_int64(), id, generation
  } nobreak {
    (end, used)
  }
  {
    frames: ReadOnlyArray::from_array(frames.to_owned()),
    input_tokens,
    maximum_new_tokens,
    encoded_bytes,
  }
}

///|
pub fn PreparedSerialPrompt::frame_count(self : Self) -> Int {
  self.frames.length()
}

///|
pub fn PreparedSerialPrompt::input_token_count(self : Self) -> Int {
  self.input_tokens
}

///|
pub fn PreparedSerialPrompt::maximum_new_tokens(self : Self) -> Int {
  self.maximum_new_tokens
}

///|
/// Retained immutable frame storage, computed during pure preparation. File
/// adapters share one aggregate budget without rescanning any frame payload.
pub fn PreparedSerialPrompt::encoded_bytes(self : Self) -> Int64 {
  self.encoded_bytes
}

///|
pub fn PreparedSerialPrompt::frame_bytes(
  self : Self,
  index : Int,
) -> Bytes raise {
  guard self.frames.get(index) is Some(bytes) else {
    fail("prompt replay index is outside its prepared frames")
  }
  bytes
}
