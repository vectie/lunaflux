///|
fn ordered_segments(text : String) -> Array[String] raise {
  let bytes = @utf8.encode(text)
  let input = FixedArray::makei(bytes.length(), i => bytes[i])
  let result : Array[String] = []
  for offset = 0; offset < bytes.length(); {
    let end = ordered_script_pretoken_end(input, bytes.length(), offset)
    assert_true(end > offset && end <= bytes.length())
    result.push(@utf8.decode(bytes[offset:end]))
    continue end
  }
  result
}

///|
test "ordered split handles unmatched runs and heterogeneous whitespace widths" {
  assert_eq(ordered_segments("\ue000\ue001abc"), ["\ue000", "\ue001abc"])
  assert_eq(ordered_segments("abc中文def"), ["abc", "中文", "def"])
  assert_eq(ordered_segments("カ・ナ"), ["カ", "・", "ナ"])
  assert_eq(ordered_segments("・\nabc"), ["・", "\n", "abc"])
  assert_eq(ordered_segments("　 abc"), ["　", " abc"])
  assert_eq(ordered_segments(" \u2000abc"), [" ", "\u2000abc"])
  assert_eq(ordered_segments(" \n  abc"), [" \n", " ", " abc"])
  assert_eq(ordered_segments("€abc"), ["€", "abc"])
  assert_eq(ordered_segments("!abcdef"), ["!abcdef"])
  assert_eq(ordered_segments("A1中文2B"), ["A", "1", "中文", "2", "B"])
  assert_true(unicode_is_symbol(0x20ac))
  assert_true(unicode_is_symbol(0x1f408))
  assert_false(unicode_is_symbol(0xe000))
  assert_false(unicode_is_symbol(0x0301))
}
