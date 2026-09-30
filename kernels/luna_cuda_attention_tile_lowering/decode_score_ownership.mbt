///|
/// CUDA-only physical refinement of the portable two-score ownership law.
/// Different 16-component groups can share a bank after the K row stride.
/// Choose the first cyclic phase whose two owners never request distinct
/// words in one bank. Subword requests within the same word are broadcasts.
fn cuda_dual_score_phase_legal(dimensions : Int, phase : Int) -> Bool {
  let groups = (dimensions + 15) / 16
  for item in 0..<groups {
    for left in 0..<16 {
      let first = left + item * 16
      if first >= dimensions {
        continue
      }
      let first_word = first / 2
      for right in 0..<16 {
        let second = right + (item + phase) % groups * 16
        if second >= dimensions {
          continue
        }
        let second_word = (dimensions + second) / 2
        if first_word % 32 == second_word % 32 && first_word != second_word {
          return false
        }
      }
    }
  }
  true
}

///|
fn cuda_dual_score_group_phase(dimensions : Int) -> Int {
  for phase in 0..<((dimensions + 15) / 16) {
    if cuda_dual_score_phase_legal(dimensions, phase) {
      return phase
    }
  }
  // Correctness never depends on finding a bank-disjoint phase. An unusual
  // unsupported-for-fast-ownership layout still preserves the numeric law.
  0
}

///|
pub fn CudaAttentionTileLowering::decode_score_group_phase(
  self : CudaAttentionTileLowering,
) -> Int {
  match self.physical_program {
    Ok(BlockwiseKeyProgram(fold, _)) if fold.rotates_score_groups() =>
      cuda_dual_score_group_phase(self.head_dimension)
    _ => 0
  }
}
