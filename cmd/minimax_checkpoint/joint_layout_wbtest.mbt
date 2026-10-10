///|
test "joint residency sums every owner before allocation" {
  let parts = [64000000000L, 10000000000L, 605429308L, 1500000000L, 4294967296L]
  let exact = for part in parts; total = 0L {
    continue total + part
  } nobreak {
    total
  }
  assert_eq(joint_sum_bytes(parts, exact), exact)
  for budget in [0L, exact - 1L] {
    try joint_sum_bytes(parts, budget) catch {
      _ => ()
    } noraise {
      _ => fail("joint residency limit was ignored")
    }
  }
  try joint_sum_bytes([9223372036854775807L, 1L], 9223372036854775807L) catch {
    _ => ()
  } noraise {
    _ => fail("overflowed aggregate budget was admitted")
  }
}

///|
test "joint coefficient parsing preserves actual F32 bits and rejects truncation" {
  let values = joint_f32_values(b"\x00\x00\x80\x3f\x00\x00\x00\x3f")
  assert_eq(values, [1.0F, 0.5F])
  for bytes in [b"", b"\x00", b"\x00\x00\x80\x7f", b"\x00\x00\xc0\x7f"] {
    try joint_f32_values(bytes) catch {
      _ => ()
    } noraise {
      _ => fail("invalid checkpoint coefficient was admitted")
    }
  }
}
