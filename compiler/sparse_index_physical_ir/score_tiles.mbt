///|
/// Partition independent candidate scores before stable selection. Each score
/// retains its complete ordered head/component fold; partitioning never makes
/// a partial reduction or changes tie order. Selection reads the published
/// score/validity region only after every tile has completed.
pub struct CandidateScorePlan {
  priv queries : Int
  priv candidates : Int
  priv width : Int
}

///|
pub fn RetainedIndexPlan::score_tiles(
  self : Self,
  width : Int,
) -> CandidateScorePlan? {
  guard width > 0 && self.candidates > width else { return None }
  let tiles = (self.candidates.to_int64() + width.to_int64() - 1L) /
    width.to_int64()
  guard tiles * self.queries.to_int64() <= 2147483647L else { return None }
  Some({ queries: self.queries, candidates: self.candidates, width, })
}

///|
pub fn CandidateScorePlan::geometry(self : Self) -> (Int, Int, Int) {
  (self.queries, self.candidates, self.width)
}

///|
pub fn CandidateScorePlan::tiles_per_query(self : Self) -> Int {
  ((self.candidates.to_int64() + self.width.to_int64() - 1L) /
  self.width.to_int64()).to_int()
}

///|
pub fn CandidateScorePlan::workgroups(self : Self) -> Int {
  self.queries * self.tiles_per_query()
}

///|
/// Refinement of the semantic index step: score publication (0), followed by
/// stable selection (1). External query/pool operands precede this subprogram;
/// its arena is the existing score/validity scratch, not additional storage.
pub fn CandidateScorePlan::program(self : Self) -> @parallel.Program raise {
  let bytes = self.queries.to_int64() * self.candidates.to_int64() * 8L
  @parallel.compile(
    rank=0,
    world_size=1,
    [@parallel.Arena::new(bytes~, initialized=false)],
    [@parallel.Buffer::new(arena=0, offset_bytes=0L, byte_count=bytes)],
    [
      @parallel.Step::new(Compute(0), [
        @parallel.Access::new(buffer=0, mode=Write),
      ]),
      @parallel.Step::new(Compute(1), [
        // Cooperative-maxima selection retires winning validity flags;
        // immutable-rank selection is a read-only subset of this effect.
        @parallel.Access::new(buffer=0, mode=ReadWrite),
      ]),
    ],
  )
}
