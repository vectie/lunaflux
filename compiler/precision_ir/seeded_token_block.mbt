///|
/// Pure seed-plus-noise token block with absolute positions following the main
/// input rows. This describes dataflow, not a model family or device schedule.
pub struct SeededTokenBlockPrecision {
  priv vocabulary : Int
  priv rows : Int
  priv main_rows : Int
  priv noise : Int
  priv seed_tokens : Int
}

///|
pub fn SeededTokenBlockPrecision::new(
  vocabulary_size~ : Int,
  block_rows~ : Int,
  maximum_main_rows~ : Int,
  noise_token_id~ : Int,
  maximum_seed_tokens? : Int = 1,
) -> Self raise PrecisionPlanError {
  guard vocabulary_size > 0 &&
    block_rows > 0 &&
    block_rows <= 1048576 &&
    maximum_main_rows > 0 &&
    maximum_main_rows <= 1048576 &&
    maximum_seed_tokens > 0 &&
    maximum_seed_tokens <= maximum_main_rows &&
    noise_token_id >= 0 &&
    noise_token_id < vocabulary_size else {
    raise IllegalPlacement
  }
  {
    vocabulary: vocabulary_size,
    rows: block_rows,
    main_rows: maximum_main_rows,
    noise: noise_token_id,
    seed_tokens: maximum_seed_tokens,
  }
}

///|
/// A vector-producing verifier supplies its last live output as the next seed.
/// Scalar-producing callers retain their existing one-cell seed contract.
pub fn SeededTokenBlockPrecision::maximum_seed_tokens(self : Self) -> Int {
  self.seed_tokens
}

///|
pub fn SeededTokenBlockPrecision::geometry(self : Self) -> (Int, Int, Int, Int) {
  (self.vocabulary, self.rows, self.main_rows, self.noise)
}

///|
pub fn SeededTokenBlockPrecision::workspace_bytes(self : Self) -> Int64 {
  20L + self.rows.to_int64() * 8L
}
