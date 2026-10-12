///|
// Initial parameter producer, norm/affine, four attention stages, residual,
// norm/affine, two MLP stages, residual. These are diagnostic frontiers of the
// existing bound launch list, not a second implementation of block arithmetic.
fn joint_block_frontiers() -> Array[(String, Int, @latent.RefinerStage)] {
  [
    ("norm1", 2, Norm1),
    ("attention-input", 3, Norm1),
    ("qkv", 4, Qkv),
    ("qk-norm", 5, QkNorm),
    ("attention", 6, Attention),
    ("projection", 7, Projection),
    ("attention-residual", 8, AttentionResidual),
    ("norm2", 9, Norm2),
    ("mlp-input", 10, Norm2),
    ("gate-up", 11, GateUp),
    ("down", 12, Down),
  ]
}

///|
test "first-block observers retire attention before overlapping MLP scratch" {
  let stages = joint_block_frontiers()
  assert_eq(stages.map(stage => stage.1), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  assert_true(stages[3] is ("qk-norm", 5, QkNorm))
  assert_true(stages[4] is ("attention", 6, Attention))
  assert_true(stages[6] is ("attention-residual", 8, AttentionResidual))
  assert_true(stages[9] is ("gate-up", 11, GateUp))
  assert_true(stages.all(stage => stage.1 < 13))
}

///|
fn joint_parameter_capture_bytes(times : Int, width : Int) -> Int64 {
  // Six AdaLN fields, three modalities, BF16; native output is field-major.
  times.to_int64() * 6L * 3L * width.to_int64() * 2L
}

///|
test "parameter observation retains every modality and unique-time field" {
  assert_eq(joint_parameter_capture_bytes(1, 5376), 193536L)
  assert_eq(joint_parameter_capture_bytes(2, 5376), 387072L)
}
