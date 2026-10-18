///|
/// Exact total-order permutation followed by the existing ordered F64
/// probability program. Compare/exchange stages are pure, disjoint matchings;
/// a device lowering must publish each stage before consuming the next one.
/// Neither the network nor the probability law depends on a model or backend.
pub struct FullVocabularySamplingProgram {
  priv values : Int32
  priv padded : Int32
  priv rows : Int32
  priv stages : ReadOnlyArray[(Int32, Int32)]
  priv final_program : StochasticProgram
}

///|
pub fn FullVocabularySamplingProgram::new(
  values~ : Int32,
  maximum_rows~ : Int32,
  maximum_workspace_bytes~ : Int64,
) -> Self? {
  // Keep signed indexing and the next-power-of-two operation representable.
  guard values > 0 && values <= 67108864 && maximum_rows > 0 else {
    return None
  }
  let padded = for n = 1; n < values; {
    continue n * 2
  } nobreak {
    n
  }
  let row_bytes = (padded.to_int64() + values.to_int64()) * 8L
  guard maximum_workspace_bytes >= row_bytes &&
    maximum_rows.to_int64() <= maximum_workspace_bytes / row_bytes else {
    return None
  }
  let stages : Array[(Int32, Int32)] = []
  for width = 2; width <= padded; width = width * 2 {
    for distance = width / 2; distance > 0; distance = distance / 2 {
      stages.push((width, distance))
    }
  }
  guard StochasticProgram::new(
      values~,
      owners=1,
      subgroup_lanes=1,
      maximum_top_k=values,
    )
    is Some(final_program) else {
    return None
  }
  Some({
    values,
    padded,
    rows: maximum_rows,
    stages: ReadOnlyArray::from_array(stages),
    final_program,
  })
}

///|
pub fn FullVocabularySamplingProgram::geometry(
  self : Self,
) -> (Int32, Int32, Int32) {
  (self.values, self.padded, self.rows)
}

///|
pub fn FullVocabularySamplingProgram::stages(
  self : Self,
) -> ReadOnlyArray[(Int32, Int32)] {
  self.stages
}

///|
pub fn FullVocabularySamplingProgram::weight_offset_bytes(self : Self) -> Int64 {
  self.padded.to_int64() * self.rows.to_int64() * 8L
}

///|
pub fn FullVocabularySamplingProgram::workspace_bytes(self : Self) -> Int64 {
  self.weight_offset_bytes() +
  self.values.to_int64() * self.rows.to_int64() * 8L
}

///|
pub fn FullVocabularySamplingProgram::final_steps(
  self : Self,
) -> ReadOnlyArray[StochasticFinalStep] {
  self.final_program.final_steps()
}
