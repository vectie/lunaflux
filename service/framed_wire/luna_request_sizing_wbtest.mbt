///|
test "Luna framed request workspace reports exact fixed cells" {
  let limits = fixture_limits()
  assert_eq(LunaFramedRequestWorkspace::required_byte_cells(limits), 4096UL)
  assert_eq(LunaFramedRequestWorkspace::required_int_cells(limits), 6UL)
  let workspace = LunaFramedRequestWorkspace::new(
    limits,
    LunaFramedRequestStepBudget::new(1),
  )
  assert_eq(workspace.storage.length().to_uint64(), 4096UL)
  assert_eq(workspace.stop_string_offsets.length().to_uint64(), 3UL)
  assert_eq(workspace.stop_string_lengths.length().to_uint64(), 3UL)
}

///|
test "Luna framed request workspace sizing rejects forged frame bounds" {
  for maximum in [-1, 0, REQUEST_HEADER_BYTES - 1, ABSOLUTE_MAX_FRAME_BYTES + 1] {
    let forged : FramedWireLimits = {
      inference: fixture_inference_limits(),
      max_frame_bytes: maximum,
    }
    try LunaFramedRequestWorkspace::required_byte_cells(forged) catch {
      InvalidLimit(FrameBytes) => ()
      _ => fail("unexpected forged byte-capacity error")
    } noraise {
      _ => fail("forged byte capacity was accepted")
    }
    try LunaFramedRequestWorkspace::required_int_cells(forged) catch {
      InvalidLimit(FrameBytes) => ()
      _ => fail("unexpected forged integer-capacity error")
    } noraise {
      _ => fail("forged integer capacity was accepted")
    }
  }
}

///|
test "Luna framed request workspace sizing checks cell multiplication" {
  assert_eq(
    luna_framed_request_checked_multiply_cells(0UL, 0xffffffffffffffffUL),
    0UL,
  )
  assert_eq(luna_framed_request_checked_multiply_cells(64UL, 2UL), 128UL)
  try
    luna_framed_request_checked_multiply_cells(0xffffffffffffffffUL, 2UL)
  catch {
    InvalidLimit(FrameBytes) => ()
    _ => fail("unexpected cell-overflow error")
  } noraise {
    _ => fail("cell multiplication overflow was accepted")
  }
}
