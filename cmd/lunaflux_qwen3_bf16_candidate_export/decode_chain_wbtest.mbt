///|
fn decode_chain_fixture() -> (
  @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier,
  DecodeChainSearch,
) raise {
  let (_, search) = extract_decode_chain_search([
    "--decode-chain-search", "8", "4096", "452,480,482", "64,128,256",
  ])
  guard search is Some(search) else { fail("missing search") }
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let baseline = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
      backend=@luna_cuda_attention_autotune.cuda_attention_backend(),
      target_id=@luna_cuda_attention_autotune.cuda_architecture_target(12, 1),
      problem=@luna_attention_strategy.AttentionCompileProblem::new(
        shape=@luna_attention_strategy.AttentionShape::new(
          query_heads=16,
          kv_heads=4,
          head_dimension=128,
          tokens_per_page=16,
        ),
        phase=Decode,
        numeric_type=AttentionBFloat16,
        query_rows=1,
        batch_rows=8,
        context_tokens=8192,
        causal=true,
        allow_alternative_softmax_reduction=true,
      ),
      capabilities=@luna_cuda_attention_autotune.cuda_sm120_decode_compiler_capabilities(),
      device_target=target,
      maximum_shared_memory_bytes=49152L,
      abi=@luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
        query_heads=16,
        key_value_heads=4,
        head_dimension=128,
        tokens_per_page=16,
        input_row_width=3072,
        page_stride_values=8192L,
        physical_page_capacity=1024L,
        maximum_query_rows=8,
        maximum_query_tokens=128,
        maximum_page_entries=8192,
      ),
      symbol="ordinary",
      maximum_source_bytes=1048576,
    ),
    [],
    explicit_candidate=452,
  ).selected()
  (prepare_decode_chain_search(search, baseline, target), search)
}

///|
test "decode chain CLI bounds and owns the explicit search domain" {
  let (rest, search) = extract_decode_chain_search([
    "before", "--decode-chain-search", "8", "4096", "452,480", "64,128", "after",
  ])
  assert_eq(rest, ["before", "after"])
  guard search is Some(search) else { fail("missing search") }
  assert_eq(search.rows, 8)
  assert_eq(search.history, 4096)
  assert_eq(search.candidates, [452, 480])
  assert_eq(search.targets, [64, 128])
  for
    suffix in [
      ["8", "4096", "452,452", "64"],
      ["8", "4096", "452", "64,64"],
      ["0", "4096", "452", "64"],
      ["8", "4096", "452", ""],
      ["8", "4096"],
    ] {
    let arguments = ["--decode-chain-search"]
    arguments.append(suffix)
    try extract_decode_chain_search(arguments) catch {
      InvalidArguments => ()
      _ => fail("unexpected failure")
    } noraise {
      _ => fail("invalid search accepted")
    }
  }
  let args = approximate_cli_base()
  args.append([
    "--decode-chain-search",
    "8",
    "4096",
    "452,480",
    "64,128",
    "--decode-chain-tuning",
    "/timings",
    "a".repeat(64),
    "gpu",
  ])
  let parsed = parse_qwen3_candidate_export_arguments(args)
  assert_true(parsed.decode_chain_search is Some(_))
  assert_true(parsed.decode_chain_tuning is Some(_))
  assert_false(parsed.bind_decode_chain)
  args.push("--bind-decode-chain")
  assert_true(parse_qwen3_candidate_export_arguments(args).bind_decode_chain)
  let no_search = approximate_cli_base()
  no_search.append(["--decode-chain-tuning", "/timings", "a".repeat(64), "gpu"])
  try parse_qwen3_candidate_export_arguments(no_search) catch {
    InvalidArguments => ()
    _ => fail("unexpected failure")
  } noraise {
    _ => fail("chain measurements accepted without search domain")
  }
  let no_observation = approximate_cli_base()
  no_observation.push("--bind-decode-chain")
  try parse_qwen3_candidate_export_arguments(no_observation) catch {
    InvalidArguments => ()
    _ => fail("unexpected binding failure")
  } noraise {
    _ => fail("binding without a measured selection")
  }
}

///|
test "selected chain keeps its lowering through serving symbol adaptation" {
  let (frontier, _) = decode_chain_fixture()
  let partitioned = frontier.variants()[0].compilation()
  let problem = partitioned.functional().plan().problem()
  let ordinary = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest::new(
      backend=@luna_cuda_attention_autotune.cuda_attention_backend(),
      target_id=@luna_cuda_attention_autotune.cuda_architecture_target(12, 1),
      problem~,
      capabilities=@luna_cuda_attention_autotune.cuda_sm120_decode_compiler_capabilities(),
      device_target=@catalog.DeviceTarget::new(
        compute_major=12,
        compute_minor=1,
        supports_bf16=true,
        supports_cublas_lt=false,
      ),
      maximum_shared_memory_bytes=49152L,
      abi=partitioned.source().abi(),
      symbol="ordinary",
      maximum_source_bytes=1048576,
    ),
    [],
    explicit_candidate=452,
  ).selected()
  let source = qwen_decode_partition_serving_source(ordinary, partitioned)
  assert_eq(source.abi(), partitioned.source().abi())
  assert_eq(source.lowering_digest(), partitioned.source().lowering_digest())
  assert_eq(source.workspace(), partitioned.source().workspace())
  assert_eq(
    source.symbols().0,
    "lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_3903",
  )
  assert_eq(
    source.symbols().1,
    "lunaflux_paged_attention_bf16_decode_split_merge_v3_ep_3904",
  )
  assert_eq(partitioned.lowering().key_value_partitions(), 2)
  let matrix = frontier
  .variants()
  .filter(v => {
    v.compilation().functional().plan().candidate().stable_id() == 480
  })[0].compilation()
  try qwen_decode_partition_serving_source(ordinary, matrix) catch {
    QwenDecodeSplitPreparation => ()
    _ => fail("unexpected arithmetic mismatch error")
  } noraise {
    _ => fail("measuring matrix output cannot widen the serving law")
  }
}

///|
test "export selection names exact complete chain and both launch domains" {
  let (frontier, search) = decode_chain_fixture()
  assert_eq(frontier.variants().length(), 9)
  let scope = decode_chain_scope(frontier, "gpu", "compiler-flags", "flags")
  let header = "luna-decode-chain-timing-v1\nscope\t" +
    scope +
    "\nworkload\tdecode\t8\t4096\n"
  let first = frontier.variants()[0].compilation()
  let winner = frontier.variants()[7].compilation()
  let rows = "chain\t\{first.digest().as_hex()}\t1000\t7\nchain\t\{winner.digest().as_hex()}\t900\t7\n"
  let selected = select_decode_chain_record(
    frontier,
    search,
    (@utf8.encode(header + rows), "gpu", "compiler-flags"),
    compiler_flags="flags",
  )
  assert_eq(selected, winner)
  assert_eq(selected.source().bytes(), winner.source().bytes())
  assert_true(
    decode_chain_launch_record(selected, search.rows).contains(
      "partial_grid=8,4,8",
    ),
  )
  assert_true(
    decode_chain_launch_record(selected, search.rows).contains(
      "merge_grid=8,16,1",
    ),
  )
  for
    invalid in [
      (header + rows).replace(
        old="workload\tdecode\t8\t4096",
        new="workload\tdecode\t4\t4096",
      ),
      (header + rows).replace(
        old="standalone-partial-merge-v1",
        new="partial-only",
      ),
      (header + rows).replace(old="\t7\n", new="\t2\n"),
      header + rows + rows,
      (header + rows).replace(old=winner.digest().as_hex(), new="0".repeat(64)),
      (header + rows).replace(old="compiler-flags", new="other-flags"),
      (header + rows).replace(old="\tflags\t", new="\tdifferent-register-cap\t"),
    ] {
    try
      select_decode_chain_record(
        frontier,
        search,
        (@utf8.encode(invalid), "gpu", "compiler-flags"),
        compiler_flags="flags",
      )
    catch {
      InvalidArguments => ()
      _ => fail("unexpected failure")
    } noraise {
      _ => fail("incomparable chain timings accepted")
    }
  }
}

///|
test "whole chain latency accepts long contexts and checks overflow" {
  assert_eq(decode_chain_latency("2000000000"), 2000000000L)
  assert_eq(decode_chain_latency("9223372036854775807"), 9223372036854775807L)
  for value in ["0", "-1", "1.5", "9223372036854775808"] {
    try decode_chain_latency(value) catch {
      InvalidArguments => ()
      _ => fail("unexpected failure")
    } noraise {
      _ => fail("invalid latency accepted")
    }
  }
}
