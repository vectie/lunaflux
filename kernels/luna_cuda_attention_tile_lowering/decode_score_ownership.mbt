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
      if fold.score_owners() == 2 {
        cuda_dual_score_group_phase(self.head_dimension)
      } else {
        cuda_owned_score_group_phase(self.head_dimension, fold.score_owners())
      }
    _ => 0
  }
}

///|
/// Refine arbitrary legal score ownership against the actual word/bank map.
/// A cyclic traversal remains a bijection regardless of the selected phase;
/// this is a layout optimization, not numerical permission inferred in CUDA.
fn cuda_owned_score_phase_legal(
  dimensions : Int,
  owners : Int,
  phase : Int,
) -> Bool {
  let lanes = 32 / owners
  let groups = (dimensions + lanes - 1) / lanes
  for item in 0..<groups {
    for left_owner in 0..<owners {
      for right_owner in (left_owner + 1)..<owners {
        for left_lane in 0..<lanes {
          let left = left_lane + (item + left_owner * phase) % groups * lanes
          if left >= dimensions {
            continue
          }
          let left_word = (left_owner * dimensions + left) / 2
          for right_lane in 0..<lanes {
            let right = right_lane +
              (item + right_owner * phase) % groups * lanes
            if right >= dimensions {
              continue
            }
            let right_word = (right_owner * dimensions + right) / 2
            if left_word % 32 == right_word % 32 && left_word != right_word {
              return false
            }
          }
        }
      }
    }
  }
  true
}

///|
fn cuda_owned_score_group_phase(dimensions : Int, owners : Int) -> Int {
  for phase in 0..<((dimensions + 32 / owners - 1) / (32 / owners)) {
    if cuda_owned_score_phase_legal(dimensions, owners, phase) {
      return phase
    }
  }
  0
}
