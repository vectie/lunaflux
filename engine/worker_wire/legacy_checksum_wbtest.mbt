///|
// Legacy frame checksum generation exists only in tests. It lets compatibility
// and structural rejection tests supply old senders' bytes without putting a
// payload scan back into production encode/decode.
fn legacy_checksum(
  bytes : FixedArray[Byte],
  length : Int,
  checksum_offset : Int,
) -> UInt {
  for index = 0, hash = 2166136261U; index < length; {
    let next = if index < checksum_offset || index >= checksum_offset + 4 {
      (hash ^ bytes[index].to_uint()) * 16777619U
    } else {
      hash
    }
    continue index + 1, next
  } nobreak {
    hash
  }
}

///|
fn frame_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  legacy_checksum(bytes, length, plan_checksum_offset)
}

///|
fn completion_frame_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  legacy_checksum(bytes, length, completion_checksum_offset)
}

///|
test "completion accepts unused checksum words without scanning payload" {
  let owner = completion_test_owner()
  let bytes = valid_completion_frame()
  for checksum in [0U, 1U, 0xffffffffU] {
    write_u32(bytes, completion_checksum_offset, checksum)
    let frame = owner.load(bytes, bytes.length())
    assert_prior_completion_is_live(frame)
  }
}
