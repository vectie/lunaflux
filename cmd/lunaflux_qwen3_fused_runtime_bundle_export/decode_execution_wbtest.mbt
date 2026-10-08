///|
test "exact decode law and typed geometry are explicit exporter inputs" {
  let raw = runtime_bundle_test_arguments("5")
  raw.append([
    "--ingress-evaluate", "full", "--decode-execution", "owned4-blockwise-fma-f32-probability-v4",
    "64", "2",
  ])
  let (args, _) = prepare_ingress_choice(raw)
  assert_true(args.blockwise_decode)
  let tagged = raw.copy()
  tagged.append([
    "--decode-symbol", "lunaflux_attention_decode_tile_compiler_v1_c467_owned4_blockwise_fma_f32_v4",
  ])
  let (tagged_args, _) = prepare_ingress_choice(tagged)
  assert_eq(
    tagged_args.decode_function_symbol,
    Some(
      "lunaflux_attention_decode_tile_compiler_v1_c467_owned4_blockwise_fma_f32_v4",
    ),
  )
  assert_eq(args.decode_execution.unwrap().storage_geometry(), (64, 2))
  assert_eq(args.decode_execution.unwrap().partition_count(), 8)
  let partitioned = raw.copy()
  partitioned.append(["--decode-partitions", "2"])
  let (partitioned_args, _) = prepare_ingress_choice(partitioned)
  assert_eq(partitioned_args.decode_execution.unwrap().partition_count(), 2)
  for count in ["0", "1", "65"] {
    let invalid = raw.copy()
    invalid.append(["--decode-partitions", count])
    try prepare_ingress_choice(invalid) catch {
      InvalidArguments => ()
      _ => fail("unexpected partition count error")
    } noraise {
      _ => fail("invalid partition count")
    }
  }
  let duplicate = raw.copy()
  duplicate.append(["--decode-partitions", "8", "--decode-partitions", "8"])
  try prepare_ingress_choice(duplicate) catch {
    InvalidArguments => ()
    _ => fail("unexpected duplicate partition error")
  } noraise {
    _ => fail("duplicate partition selection")
  }
  assert_eq(
    args.decode_execution.unwrap().numeric_law(),
    "owned4-blockwise-fma-f32-probability-v4",
  )
  raw.append(["--decode-dual-score-blockwise-f32-v2"])
  try prepare_ingress_choice(raw) catch {
    InvalidArguments => ()
    _ => fail("unexpected conflicting numerical permission error")
  } noraise {
    _ => fail("two numerical decode laws admitted")
  }
}
