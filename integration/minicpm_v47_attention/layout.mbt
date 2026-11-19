///|
pub(all) enum Slot {
  Counts
  CanvasPositions
  CausalPositions
  RowOffsets
  SequenceLengths
  PageOffsets
  PhysicalPages
  Input
  Output
  QueryWeight
  KeyWeight
  ValueWeight
  OutputWeight
  QueryNormWeight
  KeyNormWeight
  KeyCache
  ValueCache
  RawQuery
  RawKey
  RawValue
  Query
  Key
  Cosine
  Sine
  PackedQkv
  Context
  GatedContext
} derive(Debug, Eq)

///|
pub extend Slot with Eq::{equal, not_equal}

///|
pub extend Slot with @debug.Debug::{to_repr}

///|
pub fn Slot::name(self : Self) -> String {
  match self {
    Counts => "counts"
    CanvasPositions => "canvas_positions"
    CausalPositions => "causal_positions"
    RowOffsets => "row_offsets"
    SequenceLengths => "sequence_lengths"
    PageOffsets => "page_offsets"
    PhysicalPages => "physical_pages"
    Input => "input"
    Output => "output"
    QueryWeight => "query_weight"
    KeyWeight => "key_weight"
    ValueWeight => "value_weight"
    OutputWeight => "output_weight"
    QueryNormWeight => "query_norm_weight"
    KeyNormWeight => "key_norm_weight"
    KeyCache => "key_cache"
    ValueCache => "value_cache"
    RawQuery => "raw_query"
    RawKey => "raw_key"
    RawValue => "raw_value"
    Query => "query"
    Key => "key"
    Cosine => "cosine"
    Sine => "sine"
    PackedQkv => "packed_qkv"
    Context => "context"
    GatedContext => "gated_context"
  }
}

///|
pub(all) struct Region {
  slot : Slot
  workspace_offset : Int64?
  byte_count : Int64
  alignment : Int64
}

///|
fn regions(
  operators : @operators.DecoderOperators,
  compilation : Compilation,
) -> (ReadOnlyArray[Region], Int64) raise {
  let (heads, width, _, rows, _) = operators.query_rotary().geometry()
  let (kv_heads, _, _, _, _) = operators.key_rotary().geometry()
  let (_, hidden, _) = operators.full_projection(Output).geometry()
  let (sequences, _, entries) = compilation.abi().limits()
  let row = rows.to_int64()
  let qw = heads.to_int64() * width.to_int64()
  let kw = kv_heads.to_int64() * width.to_int64()
  let shape_bytes = fn(site) raise {
    let (_, columns, inner) = operators.full_projection(site).geometry()
    columns.to_int64() * inner.to_int64() * 2L
  }
  let borrowed : Array[(Slot, Int64, Int64)] = [
    (Counts, 20L, 4L),
    (CanvasPositions, row * 12L, 4L),
    (CausalPositions, row * 4L, 4L),
    (RowOffsets, (sequences + 1).to_int64() * 4L, 4L),
    (SequenceLengths, sequences.to_int64() * 4L, 4L),
    (PageOffsets, (sequences + 1).to_int64() * 4L, 4L),
    (PhysicalPages, entries.to_int64() * 4L, 4L),
    (Input, row * hidden.to_int64() * 2L, 2L),
    (Output, row * hidden.to_int64() * 2L, 2L),
    (QueryWeight, shape_bytes(QueryAndGate), 2L),
    (KeyWeight, shape_bytes(Key), 2L),
    (ValueWeight, shape_bytes(Value), 2L),
    (OutputWeight, shape_bytes(Output), 2L),
    (QueryNormWeight, width.to_int64() * 2L, 2L),
    (KeyNormWeight, width.to_int64() * 2L, 2L),
    (KeyCache, compilation.cache_bytes() / 2L, 2L),
    (ValueCache, compilation.cache_bytes() / 2L, 2L),
  ]
  let scratch : Array[(Slot, Int64)] = [
    (RawQuery, row * qw * 4L),
    (RawKey, row * kw * 2L),
    (RawValue, row * kw * 2L),
    (Query, row * qw * 2L),
    (Key, row * kw * 2L),
    (Cosine, operators.query_rotary().coefficient_bytes() / 2L),
    (Sine, operators.query_rotary().coefficient_bytes() / 2L),
    (PackedQkv, row * (qw + kw * 2L) * 2L),
    (Context, row * qw * 2L),
    (GatedContext, row * qw * 2L),
  ]
  let out : Array[Region] = borrowed.map(entry => {
    let (slot, byte_count, alignment) = entry
    { slot, workspace_offset: None, byte_count, alignment, }
  })
  let mut offset = 0L
  for (slot, bytes) in scratch {
    guard offset <= 9223372036854775792L &&
      bytes <= 9223372036854775807L - (offset + 15L) / 16L * 16L else {
      fail("attention workspace overflow")
    }
    offset = (offset + 15L) / 16L * 16L
    out.push({
      slot,
      workspace_offset: Some(offset),
      byte_count: bytes,
      alignment: 2L,
    })
    offset += bytes
  }
  (ReadOnlyArray::from_array(out), offset)
}
