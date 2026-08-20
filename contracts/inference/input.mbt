///|
/// Immutable owned token buffer admitted at the native contract boundary.
///
/// The type deliberately has no `Debug` because token IDs encode prompt
/// payload. Construction checks the collection bound before copying.
pub struct TokenBuffer {
  priv backing : LunaTokenBufferBacking
  priv token_count : Int
  priv maximum_token_id : Int
}

///|
priv enum LunaTokenBufferBacking {
  LunaOwnedTokenBacking(ReadOnlyArray[Int])
  LunaLeasedTokenBacking(LunaTokenBufferStorage, UInt64)
}

///|
/// Allocation-free scalar token read. A non-ready status carries no token ID;
/// `is_stale` distinguishes an invalidated Luna storage generation from an
/// invalid index.
#valtype
pub struct LunaTokenBufferReadStatus(Int, Int)

///|
/// Allocation-free result of checking a live token buffer against a caller's
/// token-ID bound. The cached maximum itself is never exposed.
#valtype
pub struct LunaTokenBufferBoundStatus {
  priv state : Int
}

///|
pub fn LunaTokenBufferBoundStatus::is_within_bound(
  self : LunaTokenBufferBoundStatus,
) -> Bool {
  self.state == 0
}

///|
pub fn LunaTokenBufferBoundStatus::is_out_of_range(
  self : LunaTokenBufferBoundStatus,
) -> Bool {
  self.state == 1
}

///|
pub fn LunaTokenBufferBoundStatus::is_stale(
  self : LunaTokenBufferBoundStatus,
) -> Bool {
  self.state == 2
}

///|
pub fn LunaTokenBufferReadStatus::is_ready(
  self : LunaTokenBufferReadStatus,
) -> Bool {
  self.0 == 0
}

///|
pub fn LunaTokenBufferReadStatus::is_stale(
  self : LunaTokenBufferReadStatus,
) -> Bool {
  self.0 == 2
}

///|
/// Returns the token ID for a ready status and `-1` otherwise. Token IDs are
/// contractually nonnegative, so the sentinel cannot be mistaken for data.
pub fn LunaTokenBufferReadStatus::token_id(
  self : LunaTokenBufferReadStatus,
) -> Int {
  if self.0 == 0 {
    self.1
  } else {
    -1
  }
}

///|
pub fn TokenBuffer::new(
  token_ids : ArrayView[Int],
  limits : InferenceLimits,
) -> TokenBuffer raise InferenceContractError {
  if token_ids.length() == 0 {
    raise invalid(InputToken, Empty, 0)
  }
  if token_ids.length() > limits.max_input_tokens() {
    raise invalid(InputToken, TooMany, limits.max_input_tokens())
  }
  let mut maximum_token_id = -1
  for index, token_id in token_ids {
    if token_id < 0 || token_id > limits.max_token_id() {
      raise invalid(InputToken, OutOfRange, index)
    }
    if token_id > maximum_token_id {
      maximum_token_id = token_id
    }
  }
  {
    backing: LunaOwnedTokenBacking(ReadOnlyArray::from_array(token_ids)),
    token_count: token_ids.length(),
    maximum_token_id,
  }
}

///|
pub fn TokenBuffer::length(self : TokenBuffer) -> Int {
  self.token_count
}

///|
/// Reports whether this buffer's backing generation is still readable.
pub fn TokenBuffer::is_live(self : TokenBuffer) -> Bool {
  match self.backing {
    LunaOwnedTokenBacking(tokens) =>
      self.token_count > 0 && self.token_count == tokens.length()
    LunaLeasedTokenBacking(storage, epoch) =>
      storage.is_sealed_epoch(epoch, self.token_count)
  }
}

///|
/// Authenticates this buffer's exact backing generation and compares the
/// actual cached maximum token ID against `maximum_token_id`. A negative bound
/// is out of range for every live, nonempty token buffer.
pub fn TokenBuffer::maximum_token_status(
  self : TokenBuffer,
  maximum_token_id : Int,
) -> LunaTokenBufferBoundStatus {
  if !self.is_live() {
    return { state: 2 }
  }
  if maximum_token_id < 0 || self.maximum_token_id > maximum_token_id {
    { state: 1 }
  } else {
    { state: 0 }
  }
}

///|
/// Reads one token without exposing the backing collection. Invalid indices
/// return a non-ready status; released or reused Luna storage returns stale.
pub fn TokenBuffer::token_status(
  self : TokenBuffer,
  index : Int,
) -> LunaTokenBufferReadStatus {
  match self.backing {
    LunaOwnedTokenBacking(tokens) =>
      if index < 0 || index >= self.token_count {
        LunaTokenBufferReadStatus(1, 0)
      } else {
        LunaTokenBufferReadStatus(0, tokens[index])
      }
    LunaLeasedTokenBacking(storage, epoch) => {
      if !storage.is_sealed_epoch(epoch, self.token_count) {
        return LunaTokenBufferReadStatus(2, 0)
      }
      if index < 0 || index >= self.token_count {
        return LunaTokenBufferReadStatus(1, 0)
      }
      storage.read_status(epoch, self.token_count, index)
    }
  }
}

///|
/// Reads one token with typed stale/range rejection and never exposes the
/// underlying owned or reusable collection.
pub fn TokenBuffer::token_at(
  self : TokenBuffer,
  index : Int,
) -> Int raise InferenceContractError {
  let status = self.token_status(index)
  if status.is_ready() {
    return status.token_id()
  }
  if status.is_stale() {
    raise invalid(LunaTokenStorage, LunaStorageStale, 0)
  }
  raise invalid(InputToken, OutOfRange, index)
}

///|
/// Immutable bounded UTF-8 text input. It deliberately has no `Debug`.
pub struct TextInput {
  priv text : String
  priv utf8 : Bytes
}

///|
pub fn TextInput::from_utf8(
  bytes : BytesView,
  limits : InferenceLimits,
) -> TextInput raise InferenceContractError {
  if bytes.length() == 0 {
    raise invalid(InputText, Empty, 0)
  }
  if bytes.length() > limits.max_text_bytes() {
    raise invalid(InputText, TooLong, 0)
  }
  let text = @utf8.decode(bytes) catch {
    _ => raise invalid(InputText, InvalidFormat, 0)
  }
  { text, utf8: bytes.to_owned() }
}

///|
pub fn TextInput::text(self : TextInput) -> String {
  self.text
}

///|
pub fn TextInput::utf8_bytes(self : TextInput) -> Int {
  self.utf8.length()
}

///|
/// Returns the canonical immutable UTF-8 bytes retained at validation. This
/// avoids an encode-from-UTF-16 copy before bounded Luna tokenization.
pub fn TextInput::utf8(self : TextInput) -> Bytes {
  self.utf8
}

///|
/// Canonical request input. Variant payloads can only be built by checked
/// constructors and own immutable data.
pub(all) enum Input {
  TokenIds(TokenBuffer)
  Text(TextInput)
}

///|
pub fn Input::from_token_ids(
  token_ids : ArrayView[Int],
  limits : InferenceLimits,
) -> Input raise InferenceContractError {
  TokenIds(TokenBuffer::new(token_ids, limits))
}

///|
pub fn Input::from_utf8(
  bytes : BytesView,
  limits : InferenceLimits,
) -> Input raise InferenceContractError {
  Text(TextInput::from_utf8(bytes, limits))
}

///|
fn Input::validate(
  self : Input,
  limits : InferenceLimits,
) -> Unit raise InferenceContractError {
  match self {
    TokenIds(tokens) => {
      let length = tokens.length()
      if length <= 0 {
        raise invalid(InputToken, Empty, 0)
      }
      if !tokens.is_live() {
        raise invalid(LunaTokenStorage, LunaStorageStale, 0)
      }
      if length > limits.max_input_tokens() {
        raise invalid(InputToken, TooMany, limits.max_input_tokens())
      }
      let mut actual_maximum_token_id = -1
      for index in 0..<length {
        let status = tokens.token_status(index)
        let token_id = status.token_id()
        if !status.is_ready() ||
          token_id < 0 ||
          token_id > limits.max_token_id() {
          raise invalid(InputToken, OutOfRange, index)
        }
        if token_id > actual_maximum_token_id {
          actual_maximum_token_id = token_id
        }
      }
      if actual_maximum_token_id != tokens.maximum_token_id {
        raise invalid(InputToken, Inconsistent, 0)
      }
    }
    Text(text) => {
      let utf8 = text.utf8()
      if utf8.length() == 0 {
        raise invalid(InputText, Empty, 0)
      }
      if utf8.length() > limits.max_text_bytes() {
        raise invalid(InputText, TooLong, 0)
      }
      let decoded = @utf8.decode(utf8) catch {
        _ => raise invalid(InputText, InvalidFormat, 0)
      }
      if decoded != text.text() {
        raise invalid(InputText, Inconsistent, 0)
      }
    }
  }
}
