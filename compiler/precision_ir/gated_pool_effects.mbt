///|
pub fn LearnedGatedPoolPrecision::replay_boundaries(
  self : Self,
) -> ReadOnlyArray[NumericBoundary] {
  let sizes = [..self.states, ..self.frames]
  ReadOnlyArray::from_array([
    {
      label: "projection.kv",
      buffer: 3,
      after_stage: Some(1),
      scalar: ReplayF32,
      capacity_bytes: sizes[3],
      extent: CountedRows(
        buffer=9,
        index=1,
        row_bytes=sizes[3] / self.rows.to_int64(),
      ),
    },
    {
      label: "projection.gate",
      buffer: 4,
      after_stage: Some(1),
      scalar: ReplayF32,
      capacity_bytes: sizes[4],
      extent: CountedRows(
        buffer=9,
        index=1,
        row_bytes=sizes[4] / self.rows.to_int64(),
      ),
    },
    {
      label: "pooled",
      buffer: 5,
      after_stage: Some(2),
      scalar: ReplayBf16,
      capacity_bytes: sizes[5],
      extent: CountedRows(
        buffer=9,
        index=3,
        row_bytes=self.feature_width.to_int64() * 2L,
      ),
    },
    {
      label: "normalized",
      buffer: 6,
      after_stage: Some(3),
      scalar: ReplayBf16,
      capacity_bytes: sizes[6],
      extent: CountedRows(
        buffer=9,
        index=3,
        row_bytes=self.feature_width.to_int64() * 2L,
      ),
    },
    {
      label: "next.kv",
      buffer: 0,
      after_stage: Some(2),
      scalar: ReplayF32,
      capacity_bytes: sizes[0],
      extent: Capacity,
    },
    {
      label: "next.gate",
      buffer: 1,
      after_stage: Some(2),
      scalar: ReplayF32,
      capacity_bytes: sizes[1],
      extent: Capacity,
    },
    {
      label: "next.history",
      buffer: 2,
      after_stage: Some(4),
      scalar: ReplayI32,
      capacity_bytes: sizes[2],
      extent: Capacity,
    },
    {
      label: "append",
      buffer: 9,
      after_stage: Some(0),
      scalar: ReplayI32,
      capacity_bytes: sizes[9],
      extent: Capacity,
    },
  ])
}

///|
/// Whole-capacity conservative hazards; append guards delimit initialized live
/// prefixes. This is NOT a rollback transaction: reserve can reset/poison the
/// history, pool mutates retained state, publish advances the history counter.
/// Cancellation/retry needs the enclosing owner's snapshot of ALL three state
/// buffers, or disposal of that request state. Publication alone cannot undo it.
pub fn LearnedGatedPoolPrecision::effect_program(
  self : Self,
) -> @parallel.Program raise {
  let sizes = [
    ..self.states,
    ..self.frames,
    20L,
    self.rows.to_int64() * 4L,
    4L,
    self.rows.to_int64() * self.input_width.to_int64() * 2L,
    ..self.parameters,
  ]
  let arenas = sizes.mapi((i, bytes) => {
    @parallel.Arena::new(bytes~, initialized=i < 3 || i >= 10)
  })
  let buffers = sizes.mapi((i, bytes) => {
    @parallel.Buffer::new(arena=i, offset_bytes=0L, byte_count=bytes)
  })
  fn accesses(
    entries : ArrayView[(Int, @parallel.AccessMode)],
  ) -> Array[@parallel.Access] raise {
    entries.map(entry => @parallel.Access::new(buffer=entry.0, mode=entry.1))
  }
  @parallel.compile(rank=0, world_size=1, arenas, buffers, [
    @parallel.Step::new(
      Compute(0),
      accesses([
        (10, Read),
        (11, Read),
        (12, Read),
        (2, ReadWrite),
        (9, Write),
        (8, Write),
        (7, Write),
      ]),
    ),
    @parallel.Step::new(
      Compute(1),
      accesses([
        (9, Read),
        (13, Read),
        (14, Read),
        (15, Read),
        (3, Write),
        (4, Write),
      ]),
    ),
    @parallel.Step::new(
      Compute(2),
      accesses([
        (9, Read),
        (3, Read),
        (4, Read),
        (16, Read),
        (0, ReadWrite),
        (1, ReadWrite),
        (5, Write),
      ]),
    ),
    @parallel.Step::new(
      Compute(3),
      accesses([(9, Read), (5, Read), (17, Read), (6, Write)]),
    ),
    // No direct data dependence on normalized output, but publication must
    // follow all successful output work before the caller sees completion.
    @parallel.Step::new(Compute(4), accesses([(9, Read), (2, Write)]), dependencies=[
      3,
    ]),
  ])
}
