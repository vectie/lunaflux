///|
fn converter_digest(character : Char) -> String {
  character.to_string().repeat(64)
}

///|
test "converter arguments admit one exact bounded offline route" {
  let arguments = parse_qwen3_weight_convert_arguments([
    "converter",
    "/models/qwen3",
    "config.json",
    converter_digest('a'),
    converter_digest('b'),
    "model.safetensors",
    converter_digest('c'),
    "weights.numeric.safetensors",
  ])
  assert_eq(arguments.model_root, "/models/qwen3")
  assert_eq(arguments.source_locator, "model.safetensors")
  assert_eq(arguments.output_locator, "weights.numeric.safetensors")
}

///|
test "converter arguments reject aliasing, traversal, and noncanonical digests" {
  let hostile : Array[Array[String]] = [
    [
      "converter",
      "/models/qwen3",
      "config.json",
      converter_digest('a'),
      converter_digest('b'),
      "model.safetensors",
      converter_digest('c'),
    ],
    [
      "converter",
      "/models/qwen3",
      "../config.json",
      converter_digest('a'),
      converter_digest('b'),
      "model.safetensors",
      converter_digest('c'),
      "weights.numeric.safetensors",
    ],
    [
      "converter",
      "/models/qwen3",
      "config.json",
      converter_digest('A'),
      converter_digest('b'),
      "model.safetensors",
      converter_digest('c'),
      "weights.numeric.safetensors",
    ],
    [
      "converter",
      "/models/qwen3",
      "config.json",
      converter_digest('a'),
      converter_digest('b'),
      "model.safetensors",
      converter_digest('c'),
      "model.safetensors",
    ],
    [
      "converter",
      "/models/qwen3",
      "config.json",
      converter_digest('a'),
      converter_digest('b'),
      "nested/model.safetensors",
      converter_digest('c'),
      "weights.numeric.safetensors",
    ],
  ]
  for candidate in hostile {
    try parse_qwen3_weight_convert_arguments(candidate) catch {
      InvalidArguments => ()
      _ => fail("unexpected hostile argument error")
    } noraise {
      _ => fail("hostile converter arguments were admitted")
    }
  }
}
