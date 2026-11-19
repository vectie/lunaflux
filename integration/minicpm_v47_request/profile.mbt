///|
pub using @attention {type Phase}

///|
pub suberror RequestError {
  InvalidProfile
  InvalidRequest
  InvalidLifecycle
  InvalidToken
} derive(Debug)

///|
pub extend RequestError with @debug.Debug::{to_repr}

///|
pub struct Profile {
  priv rows : Int32
  priv context : Int32
  priv page_tokens : Int32
  priv pages : Int32
  priv vocabulary : Int32
}

///|
pub fn Profile::from_program(
  program : @text.Program,
) -> Self raise RequestError {
  let (vocabulary, _, streams, rows, outputs) = program
    .text_source()
    .plan()
    .geometry()
  let attention = program.decoder_program().attention_program()
  let (sequences, attention_rows, page_entries) = attention.abi().limits()
  let (_, _, _, page_tokens) = attention.abi().shape()
  let context = attention.context_tokens()
  let pages = ((context.to_int64() + page_tokens.to_int64() - 1L) /
  page_tokens.to_int64()).to_int()
  guard streams == 1 &&
    outputs == 1 &&
    sequences == 1 &&
    rows == attention_rows &&
    page_entries >= pages else {
    raise InvalidProfile
  }
  { rows, context, page_tokens, pages, vocabulary, }
}

///|
pub fn Profile::maximum_rows(self : Self) -> Int32 {
  self.rows
}

///|
pub fn Profile::context_tokens(self : Self) -> Int32 {
  self.context
}

///|
pub fn Profile::page_capacity(self : Self) -> Int32 {
  self.pages
}

///|
pub(all) enum Port {
  Counts
  TokenIds
  CanvasPositions
  CausalPositions
  RowOffsets
  SequenceLength
  PageOffsets
  PhysicalPages
  RequestSlots
  Reset
  OutputCount
  SelectedRows
  SampledToken
} derive(Eq, Debug)

///|
pub extend Port with Eq::{equal, not_equal}

///|
pub extend Port with @debug.Debug::{to_repr}

///|
fn write_i32(bytes : FixedArray[Byte], index : Int32, value : Int32) -> Unit {
  let word = value.reinterpret_as_uint()
  for byte in 0..<4 {
    bytes[index * 4 + byte] = (word >> (byte * 8)).to_byte()
  }
}

///|
fn read_i32(bytes : FixedArray[Byte]) -> Int32 {
  (bytes[0].to_uint() |
  (bytes[1].to_uint() << 8) |
  (bytes[2].to_uint() << 16) |
  (bytes[3].to_uint() << 24)).reinterpret_as_int()
}
