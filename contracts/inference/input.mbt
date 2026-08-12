///|
/// Immutable owned token buffer admitted at the native contract boundary.
///
/// The type deliberately has no `Debug` because token IDs encode prompt
/// payload. Construction checks the collection bound before copying.
pub struct TokenBuffer {
  token_ids : ReadOnlyArray[Int]
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
  for index, token_id in token_ids {
    if token_id < 0 || token_id > limits.max_token_id() {
      raise invalid(InputToken, OutOfRange, index)
    }
  }
  { token_ids: ReadOnlyArray::from_array(token_ids) }
}

///|
pub fn TokenBuffer::token_ids(self : TokenBuffer) -> ReadOnlyArray[Int] {
  self.token_ids
}

///|
pub fn TokenBuffer::length(self : TokenBuffer) -> Int {
  self.token_ids.length()
}

///|
/// Immutable bounded UTF-8 text input. It deliberately has no `Debug`.
pub struct TextInput {
  text : String
  utf8_bytes : Int
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
  { text, utf8_bytes: bytes.length() }
}

///|
pub fn TextInput::text(self : TextInput) -> String {
  self.text
}

///|
pub fn TextInput::utf8_bytes(self : TextInput) -> Int {
  self.utf8_bytes
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
      if tokens.length() > limits.max_input_tokens() {
        raise invalid(InputToken, TooMany, limits.max_input_tokens())
      }
      for index, token_id in tokens.token_ids() {
        if token_id < 0 || token_id > limits.max_token_id() {
          raise invalid(InputToken, OutOfRange, index)
        }
      }
    }
    Text(text) =>
      if text.utf8_bytes() > limits.max_text_bytes() {
        raise invalid(InputText, TooLong, 0)
      }
  }
}
