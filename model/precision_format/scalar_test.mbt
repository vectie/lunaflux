///|
test "finite scalar decode matches format standards and signed integer ranges" {
  assert_eq(@precision_format.decode_element(E2M1, 7U), 6.0F)
  assert_eq(@precision_format.decode_element(E2M1, 9U), -0.5F)
  assert_eq(@precision_format.decode_element(E4M3, 0x7eU), 448.0F)
  assert_eq(@precision_format.decode_element(E4M3, 1U), 0.001953125F)
  assert_eq(@precision_format.decode_element(E5M2, 0x7bU), 57344.0F)
  assert_eq(@precision_format.decode_element(Fp16, 0x3c00U), 1.0F)
  assert_eq(@precision_format.decode_element(Bf16, 0x3f80U), 1.0F)
  assert_eq(@precision_format.decode_element(Signed4, 8U), -8.0F)
  assert_eq(@precision_format.decode_element(Signed8, 128U), -128.0F)
  assert_eq(@precision_format.decode_scale(Ue8M0, 127U), 1.0F)
  assert_eq(
    @precision_format.decode_scale(Ue8M0, 0U),
    Float::reinterpret_from_uint(0x00400000U),
  )
}

///|
test "all finite small float codes round trip including signed zero" {
  for
    (element, bound, invalid) in [
      (@precision_format.Element::E2M1, 16U, 16U),
      (E4M3, 256U, 127U),
      (E5M2, 256U, 124U),
      (Fp16, 65536U, 31744U),
    ] {
    for code in 0U..<bound {
      let positive = code & (bound / 2U - 1U)
      if positive >= invalid {
        continue
      }
      let value = @precision_format.decode_element(element, code)
      assert_eq(@precision_format.encode_element(element, value), code)
    }
  }
}

///|
test "ties to even and finite saturation" {
  assert_eq(@precision_format.encode_element(E2M1, 0.25F), 0U)
  assert_eq(@precision_format.encode_element(E2M1, 0.75F), 2U)
  assert_eq(@precision_format.encode_element(E2M1, 5.0F), 6U)
  assert_eq(@precision_format.encode_element(E2M1, 1000.0F), 7U)
  assert_eq(@precision_format.encode_element(Signed4, -7.5F), 8U)
  assert_eq(@precision_format.encode_element(Signed4, 2.5F), 2U)
  assert_eq(@precision_format.encode_element(Signed8, -1000.0F), 128U)
  assert_eq(@precision_format.encode_scale(Ue8M0, 1.01F), 128U)
}

///|
test "non-finite payload and scale codes fail" {
  for
    (element, code) in [
      (@precision_format.Element::E4M3, 127U),
      (E4M3, 255U),
      (E5M2, 124U),
      (Fp16, 31744U),
    ] {
    try @precision_format.decode_element(element, code) catch {
      InvalidCode(_) => ()
      _ => fail("wrong failure")
    } noraise {
      _ => fail("expected non-finite rejection")
    }
  }
  try @precision_format.decode_scale(Ue8M0, 255U) catch {
    InvalidScale => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("NaN scale accepted")
  }
}
