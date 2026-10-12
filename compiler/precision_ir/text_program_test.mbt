///|
fn text_program() -> @precision_ir.TextIoPrecision raise {
  @precision_ir.TextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=8,
    stream_count=4,
    maximum_rows=5,
    maximum_output_rows=2,
    normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=168L,
  )
}

///|
fn learned_program() -> @precision_ir.LearnedTextIoPrecision raise {
  @precision_ir.LearnedTextIoPrecision::new(
    vocabulary_size=17,
    hidden_width=8,
    stream_count=4,
    maximum_rows=5,
    maximum_output_rows=2,
    normalization_epsilon=1.0e-5,
    control_epsilon=1.0e-6,
    control_normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=248L,
  )
}

///|
test "model heads retain distinct rounding and noninterchangeable collapse" {
  let mean = text_program()
  let learned = learned_program()
  assert_eq(mean.fold(StreamUpdate).extent, 4)
  assert_eq(learned.text().fold(StreamUpdate).extent, 1)
  assert_eq(mean.fold(HeadUpdate).extent, 8)
  assert_eq(mean.fold(SquareUpdate).initial_bits, 0U)
  for op in [@precision_ir.TextExpression::Normalize, Logits] {
    assert_true(
      mean.expression(op).relation_to(learned.text().expression(op)) ==
      DifferentProgram,
    )
    try
      mean.expression(op).exact_replacement(learned.text().expression(op))
    catch {
      @precision_ir.MissingAccuracyContract => ()
      e => raise e
    } noraise {
      _ => fail("different rounding authorized as exact")
    }
  }
  let collapse = learned.collapse_program()
  assert_eq(collapse.geometry(), (4, 8, 5))
  assert_eq(collapse.fold(DotUpdate).extent, 32)
  assert_eq(collapse.fold(SquareUpdate).extent, 32)
  assert_eq(collapse.fold(CollapseUpdate).extent, 4)
  assert_true(
    collapse.expression(Affine).root() ==
    Binary(AddRN, Binary(MultiplyRN, Input(0), Input(1)), Input(2)),
  )
  assert_true(
    collapse.expression(Inverse).root() ==
    Binary(
      DivideRN,
      Float32Bits(0x3f800000U),
      Unary(SquareRoot, Binary(AddRN, Input(0), Input(1))),
    ),
  )
  try mean.fold(Logits) catch {
    _ => ()
  } noraise {
    _ => fail("publication treated as fold")
  }
  try collapse.fold(Affine) catch {
    _ => ()
  } noraise {
    _ => fail("affine treated as fold")
  }
}

///|
test "head effects and replay refer to real producers and live row counts" {
  let plans = [
    text_program().egress_program(),
    learned_program().egress_program(),
  ]
  let ports = [
    text_program().replay_boundaries(),
    learned_program().replay_boundaries(),
  ]
  for i, plan in plans {
    for stage in 1..<plan.steps().length() {
      assert_true(plan.happens_before(stage - 1, stage))
    }
    for port in ports[i] {
      assert_eq(plan.buffers()[port.buffer].byte_count(), port.capacity_bytes)
      match port.after_stage {
        Some(stage) =>
          assert_true(
            plan.steps()[stage]
            .accesses()
            .any(a => a.buffer() == port.buffer && a.mode() == Write),
          )
        None => assert_true(plan.arenas()[port.buffer].initially_initialized())
      }
      if port.extent is CountedRows(buffer~, index~, row_bytes~) {
        assert_true(row_bytes > 0L && port.capacity_bytes % row_bytes == 0L)
        assert_true(
          plan.buffers()[buffer].byte_count() >= (index + 1).to_int64() * 4L,
        )
        assert_true(
          ports[i].any(p => p.buffer == buffer && p.scalar is ReplayI32),
        )
      }
    }
  }
  // Collapse covers all live decoder rows; normalized/logits use selected rows.
  let learned = ports[1]
  assert_true(
    learned[0].extent is CountedRows(buffer=3, index=3, row_bytes=16L),
  )
  assert_true(
    learned[2].extent is CountedRows(buffer=4, index=0, row_bytes=68L),
  )
  assert_eq(learned[2].label, "logits")
}
