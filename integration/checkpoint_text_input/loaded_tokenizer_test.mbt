///|
async test "shared loaded tokenizer decodes outputs after root release without checksum scans" {
  @async.with_task_group(group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-checkpoint-decode-test")
    let directory = @fs.realpath(temporary)
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let vocabulary : Map[String, Json] = Map([])
    for byte in 0..<256 {
      let scalar = if byte is (0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
        byte
      } else if byte <= 0x20 {
        0x100 + byte
      } else if byte <= 0xa0 {
        0x121 + byte - 0x7f
      } else {
        0x143
      }
      vocabulary[scalar.to_char().unwrap().to_string()] = Json::number(
        byte.to_double(),
      )
    }
    vocabulary["hi"] = Json::number(256)
    vocabulary["<s>"] = Json::number(257)
    let document =
      $|{"version":"1.0","truncation":null,"padding":null,
      $|"normalizer":null,"post_processor":null,
      $|"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
      $|"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
      $|"added_tokens":[{"id":257,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
      $|"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,
      $|"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":\{Json::object(vocabulary).stringify()},"merges":["h i"]}}
    @fs.write_file(
      directory + "/tokenizer.json",
      document,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    let spec = @checkpoint_text_input.load_checkpoint_tokenizer(
      root,
      directory,
      "tokenizer.json",
      "a".repeat(64),
      vocabulary_size=258,
      maximum_tokens=3,
    )
    let output = @checkpoint_text_input.load_checkpoint_output_text(
      root,
      directory,
      "tokenizer.json",
      "b".repeat(64),
      "104,257,105",
      vocabulary_size=258,
      maximum_tokens=3,
    )
    let empty = @checkpoint_text_input.load_checkpoint_output_text(
      root,
      directory,
      "tokenizer.json",
      "b".repeat(64),
      "",
      vocabulary_size=258,
      maximum_tokens=3,
    )
    try
      @checkpoint_text_input.load_checkpoint_output_text(
        root,
        directory,
        "tokenizer.json",
        "b".repeat(64),
        "258",
        vocabulary_size=258,
        maximum_tokens=3,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("unknown output token accepted by shared file adapter")
    }
    root.close()
    assert_eq(output.token_ids().to_owned(), [104, 257, 105])
    assert_eq(output.decoded_bytes(), b"h<s>i")
    assert_eq(empty.token_ids().length(), 0)
    assert_eq(empty.decoded_bytes(), b"")
    assert_eq(
      spec.decode_bytes([104, 257, 105], special_policy=Preserve),
      b"h<s>i",
    )
    assert_eq(spec.decode_bytes([104, 257, 105], special_policy=Skip), b"hi")
    assert_eq(
      @checkpoint_text_input.encode_text_input(spec, b"hi", maximum_tokens=1)
      .token_ids()
      .to_owned(),
      [256],
    )
    try spec.decode_bytes([258], special_policy=Preserve) catch {
      _ => ()
    } noraise {
      _ => fail("unknown output token accepted")
    }
  })
}
