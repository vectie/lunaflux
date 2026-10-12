///|
pub(all) enum ReplayScalar {
  ReplayF32
  ReplayBf16
  ReplayI32
}

///|
/// Describes an already materialized semantic boundary, not a quality metric.
/// Capture only after producer completion. Capacity is not live extent:
/// rows are append[1] for projections, append[3] for pooled/normalized outputs.
pub struct NumericBoundary {
  label : String
  buffer : Int
  after_stage : Int
  scalar : ReplayScalar
  capacity_bytes : Int64
  live_rows_append_index : Int?
}

///|
pub fn LearnedGatedPoolPrecision::replay_boundaries(
  self : Self,
) -> ReadOnlyArray[NumericBoundary] {
  let sizes = [..self.states, ..self.frames]
  ReadOnlyArray::from_array([
    {
      label: "projection.kv",
      buffer: 3,
      after_stage: 1,
      scalar: ReplayF32,
      capacity_bytes: sizes[3],
      live_rows_append_index: Some(1),
    },
    {
      label: "projection.gate",
      buffer: 4,
      after_stage: 1,
      scalar: ReplayF32,
      capacity_bytes: sizes[4],
      live_rows_append_index: Some(1),
    },
    {
      label: "pooled",
      buffer: 5,
      after_stage: 2,
      scalar: ReplayBf16,
      capacity_bytes: sizes[5],
      live_rows_append_index: Some(3),
    },
    {
      label: "normalized",
      buffer: 6,
      after_stage: 3,
      scalar: ReplayBf16,
      capacity_bytes: sizes[6],
      live_rows_append_index: Some(3),
    },
    {
      label: "next.kv",
      buffer: 0,
      after_stage: 2,
      scalar: ReplayF32,
      capacity_bytes: sizes[0],
      live_rows_append_index: None,
    },
    {
      label: "next.gate",
      buffer: 1,
      after_stage: 2,
      scalar: ReplayF32,
      capacity_bytes: sizes[1],
      live_rows_append_index: None,
    },
    {
      label: "next.history",
      buffer: 2,
      after_stage: 4,
      scalar: ReplayI32,
      capacity_bytes: sizes[2],
      live_rows_append_index: None,
    },
    {
      label: "append",
      buffer: 9,
      after_stage: 0,
      scalar: ReplayI32,
      capacity_bytes: sizes[9],
      live_rows_append_index: None,
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
