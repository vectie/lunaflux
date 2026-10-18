///|
/// Independent output columns, not a split reduction. Every vocabulary dot
/// retains its complete input axis and numerical law. Greedy joins retain
/// global IDs; candidate transport is rank-major and independent of input rows.
pub struct VocabularyShard {
  priv vocabulary : Int
  priv rank : Int
  priv world : Int
  priv outputs : Int
  priv first : Int
  priv count : Int
  priv publication : VocabularyPublication
}

///|
pub fn VocabularyShard::new(
  vocabulary~ : Int,
  rank~ : Int,
  world~ : Int,
  maximum_outputs~ : Int,
  publication? : VocabularyPublication = GreedyCandidates,
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
pub fn VocabularyShard::columns(self : Self) -> (Int, Int) {
  (self.first, self.count)
}

///|
pub fn VocabularyShard::group(self : Self) -> (Int, Int) {
  (self.rank, self.world)
}

///|
pub fn VocabularyShard::geometry(self : Self) -> (Int, Int) {
  (self.vocabulary, self.outputs)
}

///|
pub fn VocabularyShard::candidate_width(self : Self) -> Int {
  self.outputs * 3
}
