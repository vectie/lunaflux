///|
test "all named formats pack reconstruct and bounded BF16-convert" {
  for
    format in [
      @precision_format.Format::Bf16,
      Fp16,
      F32,
      Fp8E4M3,
      Fp8E5M2,
      Int8,
      Int4,
      Fp4E2M1,
      NvFp4,
      MxFp4,
      MxFp8E4M3,
      MxFp8E5M2,
    ] {
    let rep = @precision_format.Representation::new(format~, rows=2, columns=35)
    let values = Array::makei(70, i => Float::from_int(i % 15 - 7) * 0.5F)
    let tensor = @precision_format.quantize(rep, values[:])
    let error = if format == Int4 ||
      format == Fp4E2M1 ||
      format == NvFp4 ||
      format == MxFp4 {
      0.6F
    } else {
      0.1F
    }
    for i, original in values {
      assert_true((tensor.decode(i / 35, i % 35) - original).abs() <= error)
    }
    let destination = FixedArray::make(20, b'\xaa')
    tensor.decode_bf16_chunk(32L, 8, destination, destination_offset=2)
    assert_eq(destination[0], b'\xaa')
    assert_eq(destination[18], b'\xaa')
    for i in 0..<8 {
      let code = destination[2 + 2 * i].to_uint() |
        (destination[3 + 2 * i].to_uint() << 8)
      let expected = @precision_format.encode_element(
        Bf16,
        tensor.decode((32 + i) / 35, (32 + i) % 35),
      )
      assert_eq(code, expected)
    }
  }
}

///|
test "zero tensors and partial blocks have canonical finite scales" {
  for format in [@precision_format.Format::NvFp4, MxFp4, Int4, Fp8E4M3] {
    let rep = @precision_format.Representation::new(format~, rows=3, columns=17)
    let tensor = @precision_format.quantize(rep, Array::make(51, 0.0F)[:])
    for row in 0..<3 {
      for col in 0..<17 {
        assert_eq(tensor.decode(row, col), 0.0F)
      }
    }
    assert_eq(tensor.global_scale(), 1.0F)
  }
}

///|
test "packing and scale grid are represented independently of CUDA" {
  let mx = @precision_format.Representation::new(
    format=MxFp4,
    rows=2,
    columns=33,
  )
  assert_eq(mx.row_bytes(), 17L)
  assert_eq(mx.payload_bytes(), 34L)
  assert_eq(mx.scale_shape(), (2, 2))
  assert_eq(mx.scale_index(1, 32), 3L)
  assert_eq(mx.total_bytes(), 38L)
  let nv = @precision_format.Representation::new(
    format=NvFp4,
    rows=2,
    columns=33,
  )
  assert_eq(nv.total_bytes(), 44L)
  assert_true(nv.digest() != mx.digest())
  assert_eq(nv.digest(), nv.digest())
  try
    @precision_format.PackedTensor::new(
      @precision_format.Representation::new(format=Int4, rows=1, columns=1),
      payload=b"\xf1",
      scales=b"\x00\x00\x80\x3f",
    )
  catch {
    InvalidPadding => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("noncanonical padding accepted")
  }
}
