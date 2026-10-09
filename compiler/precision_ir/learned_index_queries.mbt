///|
/// Learned low-rank index queries and head weights, before score reduction.
/// Query projection uses block-128 E4M3/UE8M0; head projection and scaling
/// publish BF16. Rotary/Hadamard/E2M1 semantics remain a composed pure plan.
pub struct LearnedIndexQueriesPrecision {
  priv hidden : Int
  priv rank : Int
  priv heads : Int
  priv transform : RotaryHadamardFp4Precision
  priv query_weight : @format.Representation
  priv frames : ReadOnlyArray[Int64]
  priv workspace : Int64
}

///|
pub fn LearnedIndexQueriesPrecision::new(
  hidden_width~ : Int,
  query_rank~ : Int,
  head_count~ : Int,
  transform : RotaryHadamardFp4Precision,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let (heads, width, _, rows, _) = transform.rotary().geometry()
  guard hidden_width > 0 &&
    hidden_width <= 65536 &&
    query_rank > 0 &&
    query_rank <= 65536 &&
    query_rank % 128 == 0 &&
    head_count > 0 &&
    head_count <= 512 &&
    head_count == heads &&
    heads.to_int64() * width.to_int64() <= 65536L &&
    width % 128 == 0 else {
    raise PrecisionPlanError::IllegalPlacement
  }
  let query_weight = @format.Representation::new(
    format=MxFp8E4M3,
    rows=heads * width,
    columns=query_rank,
    grouping=Block(rows=128, columns=128),
  )
  let weights = rows.to_int64() * heads.to_int64() * 2L
  let frames = [transform.frame_bytes()[0], weights, weights]
  guard frames.all(bytes => bytes <= 4294967294L) else {
    raise PrecisionPlanError::ArithmeticOverflow
  }
  let workspace = frames.fold(init=transform.workspace_bytes(), (total, size) => {
    total + size
  })
  guard workspace <= maximum_workspace_bytes else {
    raise PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    hidden: hidden_width,
    rank: query_rank,
    heads,
    transform,
    query_weight,
    frames: ReadOnlyArray::from_array(frames),
    workspace,
  }
}

///|
pub fn LearnedIndexQueriesPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int, Int) {
  let (_, width, _, rows, _) = self.transform.rotary().geometry()
  (self.hidden, self.rank, self.heads, width, rows)
}

///|
pub fn LearnedIndexQueriesPrecision::transform(
  self : Self,
) -> RotaryHadamardFp4Precision {
  self.transform
}

///|
pub fn LearnedIndexQueriesPrecision::query_weight(
  self : Self,
) -> @format.Representation {
  self.query_weight
}

///|
/// One global head-count scale, not the local partition head count. This owner
/// currently represents the full head domain; distributed reduction is separate.
pub fn LearnedIndexQueriesPrecision::head_scale(self : Self) -> Double {
  let (_, width, _, _, _) = self.transform.rotary().geometry()
  1.0 / width.to_double().sqrt() * (1.0 / self.heads.to_double().sqrt())
}

///|
/// Raw queries, raw head projection, scaled head weights; transform owns its
/// two separate publication buffers. All five count against one budget.
pub fn LearnedIndexQueriesPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn LearnedIndexQueriesPrecision::weight_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  ReadOnlyArray::from_array([
    self.query_weight.payload_bytes(),
    self.query_weight.scale_bytes(),
    self.hidden.to_int64() * self.heads.to_int64() * 2L,
  ])
}

///|
pub fn LearnedIndexQueriesPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}

///|
pub fn LearnedIndexQueriesPrecision::effect(_self : Self) -> NumericEffect {
  StepWrite
}
