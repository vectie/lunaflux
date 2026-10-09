///|
fn layout_inputs() -> Array[JointAudioAmpParameterRegionInput] {
  Array::makei(126, index => {
    role: expected_role(index),
    tensor_name: "tensor.\{index}",
    element_count: 1L,
    byte_offset: index.to_int64() * 64L,
  })
}

///|
test "packed AMP layout rejects missing reordered duplicate and overlapping regions" {
  let missing = layout_inputs()
  ignore(missing.pop())
  try JointAudioAmpParameterLayout::new(missing) catch {
    InvalidRegionCount => ()
    _ => fail("wrong missing rejection")
  } noraise {
    _ => fail("missing region admitted")
  }
  let reordered = layout_inputs()
  let saved = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = saved
  try JointAudioAmpParameterLayout::new(reordered) catch {
    InvalidRegionOrder(0) => ()
    _ => fail("wrong reorder rejection")
  } noraise {
    _ => fail("reordered region admitted")
  }
  let overlapping = layout_inputs()
  overlapping[1] = { ..overlapping[1], byte_offset: 0L, }
  try JointAudioAmpParameterLayout::new(overlapping) catch {
    InvalidRegionOffset(1) => ()
    _ => fail("wrong overlap rejection")
  } noraise {
    _ => fail("overlap admitted")
  }
  let duplicate = layout_inputs()
  duplicate[1] = { ..duplicate[1], tensor_name: duplicate[0].tensor_name, }
  try JointAudioAmpParameterLayout::new(duplicate) catch {
    DuplicateRegion(1) => ()
    _ => fail("wrong duplicate rejection")
  } noraise {
    _ => fail("duplicate admitted")
  }
  let overflowing = layout_inputs()
  overflowing[0] = { ..overflowing[0], element_count: 2305843009213693952L, }
  try JointAudioAmpParameterLayout::new(overflowing) catch {
    SizeOverflow => ()
    _ => fail("wrong overflow rejection")
  } noraise {
    _ => fail("overflow admitted")
  }
}
