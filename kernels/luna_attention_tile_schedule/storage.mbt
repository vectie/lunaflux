///|
pub(all) enum AttentionTileStorageRole {
  QueryStorage
  KeyStorage
  ValueStorage
  ScoreStorage
  ProbabilityStorage
  AccumulatorStorage
  PreviousScaleStorage
  TerminalMaximumStorage
  TerminalDenominatorStorage
  ValidationStorage
  QueryBoundsStorage
} derive(Debug, Eq)

///|
/// Half-open lifetimes within one ordered tile iteration: 0 = staging,
/// 1 = QK, 2 = softmax, 3 = PV, 4 = terminal fold-state export.
/// Iteration barriers make reuse between these phases explicit in lowering.
pub struct AttentionTileStorageSpan {
  priv role : AttentionTileStorageRole
  priv offset : Int64
  priv bytes : Int64
  priv begin : Int
  priv end : Int
} derive(Debug, Eq)

///|
/// Immutable storage extraction for the matrix QK/PV online-softmax rewrite.
/// No device address-space names or model-family assumptions belong here.
pub struct AttentionTileStorageLayout {
  priv spans : ReadOnlyArray[AttentionTileStorageSpan]
  priv peak_bytes : Int64
} derive(Debug, Eq)

///|
pub fn AttentionTileStorageLayout::spans(
  self : AttentionTileStorageLayout,
) -> ReadOnlyArray[AttentionTileStorageSpan] {
  self.spans
}

///|
pub fn AttentionTileStorageLayout::peak_bytes(
  self : AttentionTileStorageLayout,
) -> Int64 {
  self.peak_bytes
}

///|
pub fn AttentionTileStorageSpan::role(
  self : AttentionTileStorageSpan,
) -> AttentionTileStorageRole {
  self.role
}

///|
pub fn AttentionTileStorageSpan::range(
  self : AttentionTileStorageSpan,
) -> (Int64, Int64) {
  (self.offset, self.bytes)
}

///|
pub fn AttentionTileStorageSpan::lifetime(
  self : AttentionTileStorageSpan,
) -> (Int, Int) {
  (self.begin, self.end)
}

///|
pub fn AttentionTileStorageLayout::offset(
  self : AttentionTileStorageLayout,
  role : AttentionTileStorageRole,
) -> Int64 {
  for span in self.spans {
    if span.role == role {
      return span.offset
    }
  }
  abort("complete attention storage layout missing a role")
}

///|
fn attention_storage_align(bytes : Int64) -> Int64 {
  (bytes + 15L) / 16L * 16L
}

///|
/// Sequential live groups add their sizes; mutually exclusive lifetimes take
/// their maximum. In particular probabilities AND rescale factors coexist,
/// so their sum must fit the former key region. Terminal max/denominator reuse
/// that region only after the final PV consumer completes.
fn plan_attention_matrix_storage(
  program : @luna_attention_tile_ir.AttentionTileProgram,
) -> AttentionTileStorageLayout {
  let (query, keys, head) = program.candidate().tile().dimensions()
  let stages = program.candidate().execution_geometry().1
  let element = numeric_bytes(program.problem().numeric_type()).to_int64()
  let q = query.to_int64()
  let k = keys.to_int64()
  let d = head.to_int64()
  let query_bytes = q * d * element
  let key_bytes = stages.to_int64() * k * d * element
  let probability_bytes = q * k * 2L
  let scale_bytes = q * 4L
  let key_group_bytes = attention_storage_align(
    Int64::max(
      key_bytes,
      Int64::max(probability_bytes + scale_bytes, 2L * scale_bytes),
    ),
  )
  let key_offset = attention_storage_align(query_bytes)
  let value_offset = key_offset + key_group_bytes
  let score_offset = value_offset + attention_storage_align(key_bytes)
  let score_bytes = q * k * 4L
  let accumulator_offset = score_offset + attention_storage_align(score_bytes)
  let accumulator_bytes = q * d * 4L
  {
    spans: ReadOnlyArray::from_array([
      { role: QueryStorage, offset: 0L, bytes: query_bytes, begin: 0, end: 5 },
      {
        role: KeyStorage,
        offset: key_offset,
        bytes: key_bytes,
        begin: 0,
        end: 2,
      },
      {
        role: ValueStorage,
        offset: value_offset,
        bytes: key_bytes,
        begin: 0,
        end: 4,
      },
      {
        role: ScoreStorage,
        offset: score_offset,
        bytes: score_bytes,
        begin: 1,
        end: 3,
      },
      {
        role: ProbabilityStorage,
        offset: key_offset,
        bytes: probability_bytes,
        begin: 2,
        end: 4,
      },
      {
        role: AccumulatorStorage,
        offset: accumulator_offset,
        bytes: accumulator_bytes,
        begin: 0,
        end: 5,
      },
      {
        role: PreviousScaleStorage,
        offset: key_offset + probability_bytes,
        bytes: scale_bytes,
        begin: 2,
        end: 4,
      },
      {
        role: TerminalMaximumStorage,
        offset: key_offset,
        bytes: scale_bytes,
        begin: 4,
        end: 5,
      },
      {
        role: TerminalDenominatorStorage,
        offset: key_offset + scale_bytes,
        bytes: scale_bytes,
        begin: 4,
        end: 5,
      },
      {
        role: ValidationStorage,
        offset: score_offset + score_bytes - 4L,
        bytes: 4L,
        begin: 0,
        end: 1,
      },
      {
        role: QueryBoundsStorage,
        offset: score_offset,
        bytes: 4L,
        begin: 0,
        end: 1,
      },
    ]),
    peak_bytes: accumulator_offset + attention_storage_align(accumulator_bytes),
  }
}

///|
fn attention_storage_canonical(layout : AttentionTileStorageLayout) -> String {
  let builder = StringBuilder()
  builder <+ "storage=online-softmax-lifetimes-v1,\{layout.peak_bytes}\n"
  // Constructor order is stable; it is not a hash-map iteration order.
  for i, span in layout.spans {
    builder <+
      "storage_span=\{i},\{span.offset},\{span.bytes},\{span.begin},\{span.end}\n"
  }
  builder.to_string()
}

///|
pub fn AttentionTileSchedule::matrix_storage_layout(
  self : AttentionTileSchedule,
) -> AttentionTileStorageLayout? {
  self.matrix_storage_layout
}
