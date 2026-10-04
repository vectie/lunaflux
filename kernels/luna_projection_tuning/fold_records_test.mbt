///|
test "v4 transfer effects are independent source-bound alternatives" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let baseline = "record\t\{a}\t\{a}\t100\t3\t-\n"
  for
    (token, execution) in [
      ("serial", @compiler.SerialTransfer),
      ("prefetch", CompletedPrefetch),
      ("overlapped", OverlappedTransfer),
    ] {
    let records = parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v4\nscope\ttest\n" +
        baseline +
        "record\t\{a}\t\{b}\t50\t3\texecution:\{token}\n",
      ),
      scope="test",
    )
    assert_true(records[0].is_baseline())
    assert_false(records[1].is_baseline())
    assert_eq(records[1].sibling_execution(), Some(execution))
    assert_true(records[1].fold_choices().is_empty())
  }
  for
    (version, choice) in [
      ("v3", "execution:overlapped"),
      ("v4", "execution:unknown"),
      ("v4", "execution:overlapped,execution:serial"),
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(
          "luna-projection-fold-\{version}\nscope\ttest\n" +
          baseline +
          "record\t\{a}\t\{b}\t50\t3\t\{choice}\n",
        ),
        scope="test",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid transfer choice accepted")
    }
  }
}

///|
test "v3 row domains compose global and bounded schedules without duplicate domains" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let prefix = "luna-projection-fold-v3\nscope\ttest\nrecord\t\{a}\t\{a}\t100\t3\t-\n"
  let choices = "matrix:2:4:1:4:4:1,matrix:2:4:1:1:1:0:1:16"
  let records = parse_projection_fold_records(
    @utf8.encode(prefix + "record\t\{a}\t\{b}\t50\t3\t\{choices}\n"),
    scope="test",
  )
  assert_eq(records[1].fold_choices().length(), 2)
  assert_eq(records[1].fold_choices()[0].maximum_token_rows(), None)
  assert_eq(records[1].fold_choices()[1].maximum_token_rows(), Some(16))
  try
    parse_projection_fold_records(
      @utf8.encode(
        prefix +
        "record\t\{a}\t\{b}\t50\t3\t\{choices},matrix:2:2:1:1:1:0:1:16\n",
      ),
      scope="test",
    )
  catch {
    InvalidSnapshot => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("duplicate row domain accepted")
  }
}

///|
test "fold observations require matched baseline unique roles and sample support" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let prefix = "luna-projection-fold-v1\nscope\tdevice\ttool\n"
  let baseline = "record\t\{a}\t\{a}\t100\t3\t-\n"
  for
    body in [
      "record\t\{a}\t\{b}\t50\t3\tmatrix:2:1:1\n",
      baseline + "record\t\{a}\t\{b}\t50\t2\tmatrix:2:1:1\n",
      baseline + "record\t\{a}\t\{b}\t50\t5\tmatrix:2:1:1\n",
      baseline + "record\t\{a}\t\{b}\t50\t3\tmatrix:2:1:1,matrix:3:1:1\n",
      baseline + "record\t\{a}\t\{b}\t0\t3\tmatrix:2:1:1\n",
      baseline + baseline,
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(prefix + body),
        scope="device\ttool",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid observation accepted")
    }
  }
  let records = parse_projection_fold_records(
    @utf8.encode(
      prefix + baseline + "record\t\{a}\t\{b}\t50\t3\tmatrix:2:1:1\n",
    ),
    scope="device\ttool",
  )
  assert_eq(records.length(), 2)
  assert_eq(records[1].sample_count(), 3)
  let c = "c".repeat(64)
  let d = "d".repeat(64)
  let separate = parse_projection_fold_records(
    @utf8.encode(
      prefix +
      baseline +
      "record\t\{c}\t\{c}\t200\t5\t-\nrecord\t\{c}\t\{d}\t100\t5\tmatrix:2:1:1\n",
    ),
    scope="device\ttool",
  )
  assert_eq(separate[0].sample_count(), 3)
  assert_eq(separate[2].sample_count(), 5)
}

///|
test "v2 fold records preserve executable geometry and affine address choices" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let prefix = "luna-projection-fold-v2\nscope\ttest\nrecord\t\{a}\t\{a}\t100\t3\t-\n"
  let records = parse_projection_fold_records(
    @utf8.encode(prefix + "record\t\{a}\t\{b}\t50\t3\tmatrix:2:2:1:8:8:1\n"),
    scope="test",
  )
  assert_eq(
    records[1].fold_choices()[0],
    @compiler.ProjectionFoldChoice::new(
      MatrixFold,
      stages=2,
      transfer_tiles=2,
      fragment_stages=1,
      row_tiles=8,
      output_tiles=8,
      hoist_addresses=true,
    ),
  )
  for
    choice in [
      "matrix:2:2:1:8:8:2", "matrix:2:2:1:17:8:1", "matrix:2:2:1", "matrix:2:2:1:8:33:1",
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(prefix + "record\t\{a}\t\{b}\t50\t3\t\{choice}\n"),
        scope="test",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid v2 choice accepted")
    }
  }
}

///|
test "ownership-only fold observations are alternatives with a matched baseline" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let baseline = "record\t\{a}\t\{a}\t100\t3\t-\n"
  let prefix = "luna-projection-fold-v2\nscope\ttest\n"
  for
    (token, expected) in [
      ("retained", @compiler.RetainSiblingProducer),
      ("coowned", CoownedSiblingValues),
    ] {
    let record = "record\t\{a}\t\{b}\t50\t3\tepilogue:\{token}\n"
    let records = parse_projection_fold_records(
      @utf8.encode(prefix + baseline + record),
      scope="test",
    )
    assert_true(records[0].is_baseline())
    assert_false(records[1].is_baseline())
    assert_true(records[1].fold_choices().is_empty())
    assert_eq(records[1].sibling_ownership(), expected)
    try
      parse_projection_fold_records(@utf8.encode(prefix + record), scope="test")
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("ownership alternative accepted as baseline")
    }
  }
  for
    token in [
      "epilogue:unknown", "epilogue:retained,epilogue:coowned", "epilogue:retained,epilogue:retained",
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(
          prefix + baseline + "record\t\{a}\t\{b}\t50\t3\t\{token}\n",
        ),
        scope="test",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid ownership accepted")
    }
  }
  try
    parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v1\nscope\ttest\n" +
        baseline +
        "record\t\{a}\t\{b}\t50\t3\tepilogue:retained\n",
      ),
      scope="test",
    )
  catch {
    InvalidSnapshot => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("v1 accepted ownership grammar")
  }
}

///|
test "v2 consumer groups are immutable finite fold choices" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let prefix = "luna-projection-fold-v2\nscope\ttest\nrecord\t\{a}\t\{a}\t100\t3\t-\n"
  let records = parse_projection_fold_records(
    @utf8.encode(
      prefix +
      "record\t\{a}\t\{b}\t50\t3\tsibling:2:2:1:4:8:0:8,epilogue:coowned\n",
    ),
    scope="test",
  )
  assert_eq(
    records[1].fold_choices()[0],
    @compiler.ProjectionFoldChoice::new(
      SiblingFold,
      stages=2,
      transfer_tiles=2,
      fragment_stages=1,
      row_tiles=4,
      output_tiles=8,
      consumer_groups=8,
    ),
  )
  assert_eq(records[1].sibling_ownership(), CoownedSiblingValues)
  for
    invalid in [
      "sibling:2:2:1:4:8:0:0", "sibling:2:2:1:4:8:0:3", "sibling:2:2:1:4:8:0:64",
      "sibling:2:2:1:4:8:0:8:8",
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(prefix + "record\t\{a}\t\{b}\t50\t3\t\{invalid}\n"),
        scope="test",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid consumer geometry accepted")
    }
  }
}

///|
test "v2 transport choices require baseline and cannot be duplicated" {
  let a = "a".repeat(64)
  let b = "b".repeat(64)
  let baseline = "record\t\{a}\t\{a}\t100\t3\t-\n"
  let prefix = "luna-projection-fold-v2\nscope\ttest\n"
  for
    (token, transport) in [
      ("segmented", @compiler.SegmentedSiblingTransport),
      ("independent", IndependentSiblingTransport),
    ] {
    let records = parse_projection_fold_records(
      @utf8.encode(
        prefix + baseline + "record\t\{a}\t\{b}\t50\t3\ttransport:\{token}\n",
      ),
      scope="test",
    )
    assert_true(records[0].is_baseline())
    assert_false(records[1].is_baseline())
    assert_eq(records[1].sibling_transport(), transport)
  }
  for
    token in [
      "transport:unknown", "transport:segmented,transport:independent", "transport:segmented,transport:segmented",
    ] {
    try
      parse_projection_fold_records(
        @utf8.encode(
          prefix + baseline + "record\t\{a}\t\{b}\t50\t3\t\{token}\n",
        ),
        scope="test",
      )
    catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid transport accepted")
    }
  }
  try
    parse_projection_fold_records(
      @utf8.encode(prefix + "record\t\{a}\t\{b}\t50\t3\ttransport:segmented\n"),
      scope="test",
    )
  catch {
    InvalidSnapshot => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("transport alternative became its own baseline")
  }
  try
    parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v1\nscope\ttest\n" +
        baseline +
        "record\t\{a}\t\{b}\t50\t3\ttransport:segmented\n",
      ),
      scope="test",
    )
  catch {
    InvalidSnapshot => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("v1 accepted transport")
  }
}
