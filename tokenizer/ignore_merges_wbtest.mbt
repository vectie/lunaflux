///|
test "direct whole token lookup respects regex boundaries and UTF8" {
  let vocabulary = luna_work_vocabulary()
  vocabulary.push({ token_id: 256, piece: b"abc", kind: Normal, })
  vocabulary.push({ token_id: 257, piece: b"abc!", kind: Normal, })
  vocabulary.push({
    token_id: 258,
    piece: @utf8.encode("中文"),
    kind: Normal,
  })
  let spec = TokenizerSpec::new_qwen3_bpe(
    vocabulary,
    [],
    luna_work_digest(),
    luna_work_limits(),
    ignore_merges=true,
  )
  debug_inspect(
    spec.encode_text(
      "abc!中文",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    // The regex attaches a leading punctuation character to the next word;
    // "!中文" is not a whole vocabulary piece. Do not match its suffix.
    content="[256, 33, 228, 184, 173, 230, 150, 135]",
  )
  debug_inspect(
    spec.encode_text(
      "中文",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[258]",
  )
  debug_inspect(
    spec.encode_text(
      "abc!",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids,
    content="[256, 33]",
  )
  assert_eq(
    spec.encode_text(
      "",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids.length(),
    0,
  )
}
