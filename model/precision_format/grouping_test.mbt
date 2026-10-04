///|
test "fixed formats reject conflicting scale grouping and large sizes do not wrap" {
  for
    format in [
      @precision_format.Format::NvFp4,
      MxFp4,
      MxFp8E4M3,
      Bf16,
      Fp16,
      F32,
    ] {
    try
      @precision_format.Representation::new(
        format~,
        rows=1,
        columns=32,
        grouping=Tensor,
      )
    catch {
      InvalidGrouping => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("silently replaced grouping")
    }
  }
  let large = @precision_format.Representation::new(
    format=Int4,
    rows=2147483647,
    columns=1073741823,
    grouping=Block(rows=1, columns=1),
  )
  try large.total_bytes() catch {
    SizeOverflow => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("overflow accepted")
  }
}
