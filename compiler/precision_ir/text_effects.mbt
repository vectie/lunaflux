///|
/// Head-only ports: normalized, logits, input counts, output count, selected
/// local rows, decoder output, norm weight, head weight, sampled tokens.
/// The decoder is an external producer; embedding is NOT its substitute.
pub fn TextIoPrecision::egress_program(
  self : Self,
) -> @parallel.Program raise PrecisionPlanError {
  numeric_effects(
    [
      self.normalized_bytes,
      self.logits_bytes,
      20L,
      4L,
      self.outputs.to_int64() * 4L,
      self.residual_bytes,
      self.hidden.to_int64() * 2L,
      self.table_bytes,
      self.outputs.to_int64() * 4L,
    ],
    [false, false, true, true, true, true, true, true, false],
    [
      [(2, Read), (3, Read), (4, Read), (5, Read), (6, Read), (0, Write)],
      [(3, Read), (0, Read), (7, Read), (1, Write)],
      [(2, Read), (3, Read), (4, Read), (1, Read), (8, Write)],
    ],
  )
}

///|
/// Learned head ports: collapse, normalized, logits, input counts, output count,
/// selected local rows, decoder output, learned function/base/scale, norm/head,
/// sampled tokens. No retained KV state is modified by the head.
pub fn LearnedTextIoPrecision::egress_program(
  self : Self,
) -> @parallel.Program raise PrecisionPlanError {
  numeric_effects(
    [
      self.collapse_bytes,
      self.text.normalized_bytes,
      self.text.logits_bytes,
      20L,
      4L,
      self.text.outputs.to_int64() * 4L,
      self.input_bytes,
      self.function_bytes,
      self.base_bytes,
      4L,
      self.text.hidden.to_int64() * 2L,
      self.text.table_bytes,
      self.text.outputs.to_int64() * 4L,
    ],
    [
      false, false, false, true, true, true, true, true, true, true, true, true,
      false,
    ],
    [
      [(3, Read), (6, Read), (7, Read), (8, Read), (9, Read), (0, Write)],
      [(3, Read), (4, Read), (5, Read), (0, Read), (10, Read), (1, Write)],
      [(4, Read), (1, Read), (11, Read), (2, Write)],
      [(3, Read), (4, Read), (5, Read), (2, Read), (12, Write)],
    ],
  )
}

///|
fn numeric_effects(
  sizes : ArrayView[Int64],
  initialized : ArrayView[Bool],
  stages : ArrayView[Array[(Int, @parallel.AccessMode)]],
) -> @parallel.Program raise PrecisionPlanError {
  @parallel.compile(
    rank=0,
    world_size=1,
    sizes.mapi((i, bytes) => {
      @parallel.Arena::new(bytes~, initialized=initialized[i])
    }),
    sizes.mapi((i, bytes) => {
      @parallel.Buffer::new(arena=i, offset_bytes=0L, byte_count=bytes)
    }),
    stages.mapi((i, entries) => {
      @parallel.Step::new(
        Compute(i),
        entries.map(entry => @parallel.Access::new(buffer=entry.0, mode=entry.1)),
      )
    }),
  ) catch {
    _ => raise IllegalPlacement
  }
}

///|
pub fn TextIoPrecision::replay_boundaries(
  self : Self,
) -> ReadOnlyArray[NumericBoundary] {
  [
    {
      label: "normalized",
      buffer: 0,
      after_stage: Some(0),
      scalar: ReplayBf16,
      capacity_bytes: self.normalized_bytes,
      extent: CountedRows(
        buffer=3,
        index=0,
        row_bytes=self.hidden.to_int64() * 2L,
      ),
    },
    {
      label: "logits",
      buffer: 1,
      after_stage: Some(1),
      scalar: ReplayF32,
      capacity_bytes: self.logits_bytes,
      extent: CountedRows(
        buffer=3,
        index=0,
        row_bytes=self.vocabulary.to_int64() * 4L,
      ),
    },
    {
      label: "input.counts",
      buffer: 2,
      after_stage: None,
      scalar: ReplayI32,
      capacity_bytes: 20L,
      extent: Capacity,
    },
    {
      label: "output.count",
      buffer: 3,
      after_stage: None,
      scalar: ReplayI32,
      capacity_bytes: 4L,
      extent: Capacity,
    },
    {
      label: "selected.local_rows",
      buffer: 4,
      after_stage: None,
      scalar: ReplayI32,
      capacity_bytes: self.outputs.to_int64() * 4L,
      extent: CountedRows(buffer=3, index=0, row_bytes=4L),
    },
  ]
}

///|
pub fn LearnedTextIoPrecision::replay_boundaries(
  self : Self,
) -> ReadOnlyArray[NumericBoundary] {
  let ports = self.text
    .replay_boundaries()
    .map(port => {
      ..port,
      buffer: port.buffer + 1,
      after_stage: port.after_stage.map(stage => stage + 1),
      extent: match port.extent {
        Capacity => Capacity
        CountedRows(buffer~, index~, row_bytes~) =>
          CountedRows(buffer=buffer + 1, index~, row_bytes~)
      },
    })
  ReadOnlyArray::from_array(
    [
      {
        label: "collapsed",
        buffer: 0,
        after_stage: Some(0),
        scalar: ReplayBf16,
        capacity_bytes: self.collapse_bytes,
        extent: CountedRows(
          buffer=3,
          index=3,
          row_bytes=self.text.hidden.to_int64() * 2L,
        ),
      },
      ..ports,
    ],
  )
}
