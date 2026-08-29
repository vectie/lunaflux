///|
fn qwen3_bridge_test_spec() -> @tokenizer.TokenizerSpec {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for token_id in 0..<256 {
    vocabulary.push({
      token_id,
      piece: Bytes::makei(1, _ => token_id.to_byte()),
      kind: Normal,
    })
  }
  vocabulary.push({ token_id: 256, piece: b"\xe4\xb8", kind: Normal })
  vocabulary.push({ token_id: 257, piece: b"\xe4\xb8\xad", kind: Normal })
  vocabulary.push({ token_id: 258, piece: b"<|im_start|>", kind: Special })
  vocabulary.push({ token_id: 259, piece: b"<think>", kind: Added })
  try! @tokenizer.TokenizerSpec::new_qwen3_bpe(
    vocabulary,
    [
      { left_token_id: 0xe4, right_token_id: 0xb8, result_token_id: 256 },
      { left_token_id: 256, right_token_id: 0xad, result_token_id: 257 },
    ],
    @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 4096,
      max_output_tokens: 256,
      max_decoded_bytes: 4096,
      max_vocab_entries: 260,
      max_merge_rules: 8,
      max_token_bytes: 32,
      max_special_tokens: 8,
    },
  )
}

///|
test "Qwen incremental decode preserves merges specials and split UTF-8" {
  let spec = qwen3_bridge_test_spec()
  assert_eq(
    spec.encode_text(
      "中",
      special_policy=AsOrdinary,
      output_limit=8,
      truncation=RejectOverflow,
    ).token_ids,
    [257],
  )
  let decode = try! Qwen3IncrementalDetokenizer::new(spec)
  assert_eq(try! decode.advance(0xe4), b"")
  assert_eq(try! decode.advance(0xb8), b"")
  assert_eq(try! decode.advance(0xad), b"\xe4\xb8\xad")
  assert_eq(try! decode.advance(258), b"")
  assert_eq(try! decode.advance(259), b"<think>")
  assert_eq(try! decode.finish(), b"")
}

///|
test "Qwen incremental decode uses maximal invalid UTF-8 subsequences" {
  let decode = try! Qwen3IncrementalDetokenizer::new(qwen3_bridge_test_spec())
  assert_eq(try! decode.advance(0xe2), b"")
  assert_eq(try! decode.advance(0x82), b"")
  assert_eq(try! decode.advance(0x41), b"\xef\xbf\xbdA")
  assert_eq(try! decode.advance(0x80), b"\xef\xbf\xbd")
  assert_eq(try! decode.finish(), b"")
}

///|
test "Qwen incremental decode flushes one replacement for a trailing scalar" {
  let decode = try! Qwen3IncrementalDetokenizer::new(qwen3_bridge_test_spec())
  assert_eq(try! decode.advance(0xf0), b"")
  assert_eq(try! decode.advance(0x9f), b"")
  assert_eq(try! decode.finish(), b"\xef\xbf\xbd")
  try decode.advance(0x41) catch {
    DetokenizerFinished => ()
    _ => fail("unexpected closed detokenizer error")
  } noraise {
    _ => fail("closed detokenizer advanced")
  }
}

///|
test "Qwen token SSE escapes hostile JSON text without changing token IDs" {
  let sse = Qwen3TokenSseBuffer::new(max_text_bytes=64)
  let text : Bytes = [b'"', b'\\', b'\n', b'\t', b'\x00', 0xe4, 0xb8, 0xad]
  assert_eq(
    @utf8.decode(sse.stage(257, text)),
    "data: {\"schema\":\"lunaflux.benchmark-token.v1\",\"token_id\":257,\"text\":\"\\\"\\\\\\n\\t\\u0000中\"}\n\n",
  )
  assert_eq(
    @utf8.decode(sse.terminal(b"\xef\xbf\xbd")),
    "data: {\"schema\":\"lunaflux.benchmark-terminal.v1\",\"text\":\"�\"}\n\ndata: [DONE]\n\n",
  )
}
