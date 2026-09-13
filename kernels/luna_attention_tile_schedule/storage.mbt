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
  AccumulatorLayoutProbeStorage
} derive(Debug, Eq)

///|
/// Half-open lifetimes: -1 = layout prologue, 0 = staging,
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
  if attention_fragment_ownership(
      program.candidate().compute(),
      query_rows=query,
      key_rows=keys,
      head_columns=head,
      subgroups=program.candidate().execution_geometry().0,
    ) ==
    QueryOwnedFragments {
    return plan_query_owned_storage(query, keys, head, stages)
  }
  let element = numeric_bytes(program.problem().numeric_type()).to_int64()
  let q = query.to_int64()
  let k = keys.to_int64()
  let d = head.to_int64()
  let query_bytes = q * d * element
  let key_bytes = stages.to_int64() * k * d * element
  let probability_bytes = q * k * 2L
  let scale_bytes = q * 4L
  // A lookahead transfer overlaps the complete current fold. Its keys cannot
  // share storage with current probabilities, and its failure flag cannot
  // alias scores. The synchronous layout retains its shorter lifetimes.
  let overlaps_fold = program.candidate().memory() == AttentionAsyncStagedShared
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
  let accumulator_bytes = q * d * 4L
  let resident = output_fold_storage(
      program.candidate().compute(),
      query,
      head,
      program.candidate().execution_geometry().0,
    ) ==
    SubgroupResidentOutputFold
  let fold_end = score_offset + attention_storage_align(score_bytes)
  // The register-carried output needs shared storage only before staging and
  // after the final fold. Reuse the input prefix during these disjoint phases.
  let accumulator_offset = if resident { 0L } else { fold_end }
  let output_end = if resident {
    fold_end
  } else {
    accumulator_offset + attention_storage_align(accumulator_bytes)
  }
  let probability_offset = if overlaps_fold { output_end } else { key_offset }
  let validation_offset = if overlaps_fold {
    probability_offset +
    attention_storage_align(probability_bytes + scale_bytes)
  } else {
    score_offset + score_bytes - 4L
  }
  let spans : Array[AttentionTileStorageSpan] = [
    {
      role: QueryStorage,
      offset: 0L,
      bytes: query_bytes,
      begin: 0,
      end: if resident {
        4
      } else {
        5
      },
    },
    {
      role: KeyStorage,
      offset: key_offset,
      bytes: key_bytes,
      begin: 0,
      end: if overlaps_fold {
        4
      } else {
        2
      },
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
      offset: probability_offset,
      bytes: probability_bytes,
      begin: 2,
      end: 4,
    },
    {
      role: AccumulatorStorage,
      offset: accumulator_offset,
      bytes: accumulator_bytes,
      begin: if resident {
        4
      } else {
        0
      },
      end: 5,
    },
    {
      role: PreviousScaleStorage,
      offset: probability_offset + probability_bytes,
      bytes: scale_bytes,
      begin: 2,
      end: 4,
    },
    {
      role: TerminalMaximumStorage,
      offset: if resident {
        score_offset
      } else {
        key_offset
      },
      bytes: scale_bytes,
      begin: 4,
      end: 5,
    },
    {
      role: TerminalDenominatorStorage,
      offset: (if resident { score_offset } else { key_offset }) + scale_bytes,
      bytes: scale_bytes,
      begin: 4,
      end: 5,
    },
    {
      role: ValidationStorage,
      offset: validation_offset,
      bytes: 4L,
      begin: 0,
      end: if overlaps_fold {
        4
      } else {
        1
      },
    },
    {
      role: QueryBoundsStorage,
      offset: score_offset,
      bytes: 4L,
      begin: 0,
      end: 1,
    },
  ]
  if resident {
    spans.push({
      role: AccumulatorLayoutProbeStorage,
      offset: accumulator_offset,
      bytes: accumulator_bytes,
      begin: -1,
      end: 0,
    })
  }
  {
    spans: ReadOnlyArray::from_array(spans),
    peak_bytes: if overlaps_fold {
      validation_offset + 16L
    } else {
      output_end
    },
  }
}

///|
fn plan_query_owned_storage(
  query : Int,
  keys : Int,
  head : Int,
  stages : Int,
) -> AttentionTileStorageLayout {
  let q = query.to_int64() * head.to_int64() * 2L
  let kv = keys.to_int64() * head.to_int64() * stages.to_int64() * 2L
  {
    spans: [
      { role: QueryStorage, offset: 0L, bytes: q, begin: 0, end: 4, },
      { role: KeyStorage, offset: q, bytes: kv, begin: 0, end: 4, },
      { role: ValueStorage, offset: q + kv, bytes: kv, begin: 0, end: 4, },
      {
        role: ValidationStorage,
        offset: q + 2L * kv,
        bytes: 4L,
        begin: 0,
        end: 4,
      },
      {
        role: QueryBoundsStorage,
        offset: q + 2L * kv + 4L,
        bytes: 4L,
        begin: 0,
        end: 4,
      },
    ],
    peak_bytes: q + 2L * kv + 16L,
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
