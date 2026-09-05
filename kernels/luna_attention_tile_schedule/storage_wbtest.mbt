///|
test "online softmax storage never overlaps simultaneously live values" {
  for head_dimension in [16, 32, 64, 128, 256] {
    for candidate_id in [310, 311, 312, 313, 314, 315] {
      let program = schedule_test_program(head_dimension~, candidate_id~)
      let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
        program,
      )
      let schedule = schedule_rewritten_attention_tile_program(
        program, optimization,
      )
      guard schedule.matrix_storage_layout() is Some(layout) else {
        fail("matrix rewrite missing its storage extraction")
      }
      assert_eq(layout.peak_bytes(), schedule.execution_geometry().3)
      assert_eq(
        layout,
        schedule_rewritten_attention_tile_program(program, optimization)
        .matrix_storage_layout()
        .unwrap(),
      )
      assert_eq(layout.spans().length(), 11)
      for i, left in layout.spans() {
        let (offset, bytes) = left.range()
        assert_true(
          offset >= 0L && bytes > 0L && offset + bytes <= layout.peak_bytes(),
        )
        assert_eq(offset % 4L, 0L)
        if left.role()
          is (QueryStorage
          | KeyStorage
          | ValueStorage
          | ProbabilityStorage
          | AccumulatorStorage) {
          assert_eq(offset % 16L, 0L)
        }
        for j, right in layout.spans() {
          if i >= j {
            continue
          }
          let (other_offset, other_bytes) = right.range()
          let (begin, end) = left.lifetime()
          let (other_begin, other_end) = right.lifetime()
          let simultaneous = begin < other_end && other_begin < end
          if simultaneous {
            assert_true(
              offset + bytes <= other_offset ||
              other_offset + other_bytes <= offset,
            )
          }
        }
      }
    }
  }
}

///|
test "wide query storage includes probability AND rescale bytes before V" {
  let program = schedule_test_program(head_dimension=32, candidate_id=315)
  let optimization = @luna_attention_tile_optimizer.optimize_attention_tile_program(
    program,
  )
  let schedule = schedule_rewritten_attention_tile_program(
    program, optimization,
  )
  let layout = schedule.matrix_storage_layout().unwrap()
  assert_eq(layout.offset(KeyStorage), 4096L)
  assert_eq(layout.offset(ProbabilityStorage), 4096L)
  assert_eq(layout.offset(PreviousScaleStorage), 12288L)
  // Former K tile alone occupied 4096 bytes. P and scales require 8448.
  assert_eq(layout.offset(ValueStorage), 12544L)
  assert_eq(layout.peak_bytes(), 41216L)
  assert_true(
    @utf8.decode(schedule.canonical_bytes()).contains(
      "storage=online-softmax-lifetimes-v1,41216",
    ),
  )
}

///|
test "equal query and head widths reserve a separate rescale tail" {
  let program = schedule_test_program(head_dimension=64, candidate_id=315)
  let layout = plan_attention_matrix_storage(program)
  assert_eq(layout.offset(KeyStorage), 8192L)
  assert_eq(layout.offset(PreviousScaleStorage), 16384L)
  assert_eq(layout.offset(ValueStorage), 16640L)
  assert_eq(layout.peak_bytes(), 57600L)
}

///|
test "plain schedule does not claim the optimized lifetime layout" {
  let program = schedule_test_program()
  assert_eq(
    schedule_attention_tile_program(program).matrix_storage_layout(),
    None,
  )
}
