///|
test "a fresh or non-egress stage cannot expose vocabulary output" {
  let stage = unallocated_stage()
  for prepared in [false, true] {
    if prepared {
      stage.state = Prepared
    }
    try stage.borrow_text_logits() catch {
      @device.DeviceError::InvalidArgument(StreamingTransfer) => ()
      error => raise error
    } noraise {
      _ => fail("unallocated partial stage exposed head logits")
    }
  }
  stage.close()
}
