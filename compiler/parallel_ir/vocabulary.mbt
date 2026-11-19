///|
/// Independent output columns, not a split reduction. Every vocabulary dot
/// retains its complete input axis and numerical law. Greedy joins retain
/// global IDs; candidate transport is rank-major and independent of input rows.
pub struct VocabularyShard {
  priv vocabulary : Int32
  priv rank : Int32
  priv world : Int32
  priv outputs : Int32
  priv first : Int32
  priv count : Int32
  priv publication : VocabularyPublication
}

///|
pub fn VocabularyShard::new(
  vocabulary~ : Int32,
  rank~ : Int32,
  world~ : Int32,
  maximum_outputs~ : Int32,
  publication~ : VocabularyPublication = GreedyCandidates,
) -> Self raise ParallelError {
  guard vocabulary > 0 &&
    vocabulary <= 16777216 &&
    world > 1 &&
    world <= vocabulary &&
    rank >= 0 &&
    rank < world &&
    maximum_outputs > 0 &&
    maximum_outputs.to_int64() * world.to_int64() * 12L <= 2147483647L else {
    raise Capacity
  }
  let first = (vocabulary.to_int64() * rank.to_int64() / world.to_int64()).to_int()
  let end = (vocabulary.to_int64() * (rank + 1).to_int64() / world.to_int64()).to_int()
  if publication == FullLogits {
    guard maximum_outputs.to_int64() *
      ((vocabulary - 1) / world + 1).to_int64() *
      world.to_int64() *
      4L <=
      2147483647L else {
      raise Capacity
    }
  }
  {
    vocabulary,
    rank,
    world,
    outputs: maximum_outputs,
    first,
    count: end - first,
    publication,
  }
}

///|
pub fn VocabularyShard::columns(self : Self) -> (Int32, Int32) {
  (self.first, self.count)
}

///|
pub fn VocabularyShard::group(self : Self) -> (Int32, Int32) {
  (self.rank, self.world)
}

///|
pub fn VocabularyShard::geometry(self : Self) -> (Int32, Int32) {
  (self.vocabulary, self.outputs)
}

///|
pub fn VocabularyShard::candidate_width(self : Self) -> Int32 {
  self.outputs * 3
}
