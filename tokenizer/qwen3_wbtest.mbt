///|
fn qwen3_test_spec() -> TokenizerSpec raise TokenizerError {
  let vocabulary : Array[VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  let pieces : Array[(Bytes, TokenKind)] = [
    (b"ca", Normal),
    (b"can", Normal),
    (b"'t", Normal),
    (b"can't", Normal),
    (b"he", Normal),
    (b"hel", Normal),
    (b"hell", Normal),
    (b"hello", Normal),
    (b"hello,", Normal),
    (b"20", Normal),
    (b"202", Normal),
    (b"2026", Normal),
    (b"Ca", Normal),
    (b"Caf", Normal),
    (b"Caf\xc3", Normal),
    (b"Caf\xc3\xa9", Normal),
    (b"<think>", Added),
    (b"<|im_start|>", Special),
    (b"</think>", Added),
  ]
  for entry in pieces {
    let (piece, kind) = entry
    vocabulary.push({ token_id: vocabulary.length(), piece, kind })
  }
  TokenizerSpec::new_qwen3_bpe(
    vocabulary,
    [
      { left_token_id: 99, right_token_id: 97, result_token_id: 256 },
      { left_token_id: 256, right_token_id: 110, result_token_id: 257 },
      { left_token_id: 39, right_token_id: 116, result_token_id: 258 },
      { left_token_id: 257, right_token_id: 258, result_token_id: 259 },
      { left_token_id: 104, right_token_id: 101, result_token_id: 260 },
      { left_token_id: 260, right_token_id: 108, result_token_id: 261 },
      { left_token_id: 261, right_token_id: 108, result_token_id: 262 },
      { left_token_id: 262, right_token_id: 111, result_token_id: 263 },
      { left_token_id: 263, right_token_id: 44, result_token_id: 264 },
      { left_token_id: 50, right_token_id: 48, result_token_id: 265 },
      { left_token_id: 265, right_token_id: 50, result_token_id: 266 },
      { left_token_id: 266, right_token_id: 54, result_token_id: 267 },
      { left_token_id: 67, right_token_id: 97, result_token_id: 268 },
      { left_token_id: 268, right_token_id: 102, result_token_id: 269 },
      { left_token_id: 269, right_token_id: 195, result_token_id: 270 },
      { left_token_id: 270, right_token_id: 169, result_token_id: 271 },
    ],
    TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 1024,
      max_output_tokens: 1024,
      max_decoded_bytes: 4096,
      max_vocab_entries: 512,
      max_merge_rules: 128,
      max_token_bytes: 64,
      max_special_tokens: 32,
    },
  )
}

///|
fn qwen3_reference_placeholder(token_id : Int) -> Bytes {
  Bytes::from_array([
    (0xff).to_byte(),
    (0xfe).to_byte(),
    ((token_id >> 24) & 0xff).to_byte(),
    ((token_id >> 16) & 0xff).to_byte(),
    ((token_id >> 8) & 0xff).to_byte(),
    (token_id & 0xff).to_byte(),
  ])
}

///|
fn qwen3_join_reference_piece(left : Bytes, right : Bytes) -> Bytes {
  let joined : Array[Byte] = Array::new(capacity=left.length() + right.length())
  for byte in left {
    joined.push(byte)
  }
  for byte in right {
    joined.push(byte)
  }
  Bytes::from_array(joined)
}

///|
fn qwen3_reference_spec() -> TokenizerSpec raise TokenizerError {
  let vocabulary : Array[VocabularyEntry] = Array::new(capacity=151669)
  for token_id in 0..<151669 {
    vocabulary.push({
      token_id,
      piece: qwen3_reference_placeholder(token_id),
      kind: Normal,
    })
  }
  for byte_value in 0..<256 {
    let token_id = if byte_value >= 33 && byte_value <= 126 {
      byte_value - 33
    } else if byte_value >= 161 && byte_value <= 172 {
      byte_value - 67
    } else if byte_value >= 174 {
      byte_value - 68
    } else if byte_value <= 32 {
      byte_value + 188
    } else if byte_value <= 160 {
      byte_value + 94
    } else {
      255
    }
    vocabulary[token_id] = {
      token_id,
      piece: Bytes::from_array([byte_value.to_byte()]),
      kind: Normal,
    }
  }
  let merge_ids = qwen3_reference_merge_ids()
  let merges : Array[MergeRule] = Array::new(capacity=merge_ids.length())
  for entry in merge_ids {
    let (left_token_id, right_token_id, result_token_id) = entry
    let piece = qwen3_join_reference_piece(
      vocabulary[left_token_id].piece,
      vocabulary[right_token_id].piece,
    )
    vocabulary[result_token_id] = {
      token_id: result_token_id,
      piece,
      kind: Normal,
    }
    merges.push({ left_token_id, right_token_id, result_token_id })
  }
  vocabulary[151644] = {
    token_id: 151644,
    piece: b"<|im_start|>",
    kind: Special,
  }
  vocabulary[151645] = { token_id: 151645, piece: b"<|im_end|>", kind: Special }
  vocabulary[151667] = { token_id: 151667, piece: b"<think>", kind: Added }
  vocabulary[151668] = { token_id: 151668, piece: b"</think>", kind: Added }
  TokenizerSpec::new_qwen3_bpe(
    vocabulary,
    merges,
    TokenizerDigest::parse("fedcba9876543210".repeat(4)),
    {
      max_input_bytes: 1024,
      max_output_tokens: 1024,
      max_decoded_bytes: 4096,
      max_vocab_entries: 151669,
      max_merge_rules: 512,
      max_token_bytes: 64,
      max_special_tokens: 32,
    },
  )
}

///|
fn qwen3_reference_encode(
  spec : TokenizerSpec,
  text : String,
) -> Array[Int] raise TokenizerError {
  spec.encode_text(
    text,
    special_policy=Allow,
    output_limit=1024,
    truncation=RejectOverflow,
  ).token_ids
}

///|
test "Qwen3 contractions numbers punctuation and whitespace are isolated" {
  let spec = qwen3_test_spec()
  let result = spec.encode_text(
    "hello, I can't 2026.\n",
    special_policy=Allow,
    output_limit=64,
    truncation=RejectOverflow,
  )
  debug_inspect(
    result.token_ids,
    content="[263, 44, 32, 73, 32, 257, 258, 32, 50, 48, 50, 54, 46, 10]",
  )
  assert_false(result.token_ids.contains(259))
  assert_false(result.token_ids.contains(264))
  assert_false(result.token_ids.contains(267))
  inspect(
    spec.decode_text(result.token_ids, special_policy=Preserve),
    content=(
      #|hello, I can't 2026.
      #|
    ),
  )
}

///|
test "Qwen3 NFC normalization and Unicode are deterministic" {
  let spec = qwen3_test_spec()
  let decomposed = spec.encode_text(
    "Cafe\u{0301} 与 café",
    special_policy=Allow,
    output_limit=64,
    truncation=RejectOverflow,
  )
  let composed = spec.encode_text(
    "Café 与 café",
    special_policy=Allow,
    output_limit=64,
    truncation=RejectOverflow,
  )
  assert_eq(decomposed.token_ids, composed.token_ids)
  assert_eq(decomposed.token_ids[0], 271)
  inspect(
    spec.decode_text(decomposed.token_ids, special_policy=Preserve),
    content="Café 与 café",
  )
}

///|
test "Qwen3 Unicode 17 Letter categories and NFC cover ordinary scripts" {
  assert_eq(qwen3_unicode_letter_ranges.length(), 684 * 6)
  assert_eq(qwen3_unicode_combining_ranges.length(), 403 * 7)
  assert_eq(qwen3_unicode_decomposition_index.length(), 2081 * 6)
  assert_eq(qwen3_unicode_decomposition_scalars.length(), 3127 * 3)
  assert_eq(qwen3_unicode_composition_pairs.length(), 961 * 9)
  assert_eq(qwen3_canonical_combining_class(0x0300), 230)
  assert_eq(qwen3_canonical_combining_class(0x0315), 232)
  assert_eq(qwen3_canonical_composition(0x0065, 0x0301), Some(0x00e9))
  let spec = qwen3_test_spec()
  for
    scalar in [
      '中'.to_int(),
      '繁'.to_int(),
      '日'.to_int(),
      '한'.to_int(),
      'α'.to_int(),
      'ع'.to_int(),
      'ह'.to_int(),
    ] {
    assert_true(qwen3_is_letter(scalar))
  }
  assert_false(qwen3_is_letter('🙂'.to_int()))
  assert_false(qwen3_is_letter('—'.to_int()))
  assert_false(qwen3_is_letter('\u{0301}'.to_int()))
  let cases = [
    ("A\u{0315}\u{0300}", "À\u{0315}"),
    ("Cafe\u{0301} A\u{030A}ngstro\u{0308}m", "Café Ångström"),
    ("각", "각"),
    ("你好，世界", "你好，世界"),
    ("繁體中文", "繁體中文"),
    ("日本語かなカナ", "日本語かなカナ"),
    ("한글 테스트", "한글 테스트"),
    ("Ελληνικά", "Ελληνικά"),
    ("العربية", "العربية"),
    ("हिन्दी", "हिन्दी"),
    ("🙂🚀—「」", "🙂🚀—「」"),
  ]
  for entry in cases {
    let (input, expected) = entry
    let encoded = spec.encode_text(
      input,
      special_policy=Allow,
      output_limit=256,
      truncation=RejectOverflow,
    )
    assert_eq(
      spec.decode_text(encoded.token_ids, special_policy=Preserve),
      expected,
    )
  }
}

///|
test "Qwen3 rejects malformed UTF-8 without rejecting valid scalars" {
  let spec = qwen3_test_spec()
  for input in [b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80"] {
    try
      spec.encode_bytes(
        input,
        special_policy=Allow,
        output_limit=64,
        truncation=RejectOverflow,
      )
    catch {
      InvalidUtf8Input(_) => ()
      _ => fail("unexpected malformed UTF-8 error")
    } noraise {
      _ => fail("malformed UTF-8 was accepted")
    }
  }
  let valid = qwen3_reference_encode(spec, "\u{1fae8}")
  assert_eq(spec.decode_text(valid, special_policy=Preserve), "\u{1fae8}")
}

///|
test "Qwen3 atomic added and special tokens retain distinct decode policy" {
  let spec = qwen3_test_spec()
  let result = spec.encode_text(
    "<|im_start|>hello<think>hello</think>",
    special_policy=Allow,
    output_limit=64,
    truncation=RejectOverflow,
  )
  debug_inspect(result.token_ids, content="[273, 263, 272, 263, 274]")
  inspect(
    spec.decode_text(result.token_ids, special_policy=Skip),
    content="hello<think>hello</think>",
  )
}

///|
test "Qwen3 pinned Transformers corpus matches exact IDs and decode" {
  let spec = qwen3_reference_spec()
  let cases : Array[(String, Array[Int], String)] = [
    ("Hello, Qwen!", [9707, 11, 1207, 16948, 0], "Hello, Qwen!"),
    (
      "I can't believe it's 2026.",
      [40, 646, 944, 4411, 432, 594, 220, 17, 15, 17, 21, 13],
      "I can't believe it's 2026.",
    ),
    (
      "Cafe\u{0301} 与 café",
      [34, 2577, 963, 220, 57218, 51950],
      "Café 与 café",
    ),
    (
      "line one\nline two\n",
      [1056, 825, 198, 1056, 1378, 198],
      "line one\nline two\n",
    ),
    (
      "  spaces\tand\r\nnewlines  ",
      [220, 12621, 52477, 319, 931, 7969, 256],
      "  spaces\tand\r\nnewlines  ",
    ),
    (
      "fn main { println(\"MoonBit\") }",
      [8822, 1887, 314, 13751, 445, 76019, 8344, 899, 335],
      "fn main { println(\"MoonBit\") }",
    ),
    (
      "<|im_start|>user\nhello<|im_end|>\n",
      [151644, 872, 198, 14990, 151645, 198],
      "<|im_start|>user\nhello<|im_end|>\n",
    ),
  ]
  for entry in cases {
    let (text, expected_ids, expected_decode) = entry
    let actual_ids = qwen3_reference_encode(spec, text)
    assert_eq(actual_ids, expected_ids)
    assert_eq(
      spec.decode_text(actual_ids, special_policy=Preserve),
      expected_decode,
    )
  }
}

///|
test "Qwen3 pinned chat template IDs match the reference tokenizer" {
  let spec = qwen3_reference_spec()
  let rendered = "<|im_start|>user\nExplain in one short sentence why deterministic inference matters.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  assert_eq(qwen3_reference_encode(spec, rendered), [
    151644, 872, 198, 840, 20772, 304, 825, 2805, 11652, 3170, 72349, 44378, 12850,
    13, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
  ])
}
