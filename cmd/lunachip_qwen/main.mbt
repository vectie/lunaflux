///|
/// Experimental diagnostic CLI. The simulator is a qualification tool, not a
/// production request scheduler. Checkpoints are range-read, never payload-hashed.
async fn main {
  let args = @env.args()[1:]
  let (args, clock_report) = match args {
    [
      "benchmark-stream-unary"
      | "benchmark-warm"
      | "generate-resident"
      | "generate-warm"
      | "benchmark-program-bursts"
      | "benchmark-chain"
      | "benchmark-fast"
      | "benchmark",
      ..,
      report,
    ] if args.length() == 10 => (args[:9], Some(report))
    _ => (args, None)
  }
  let (
    mode,
    contract,
    model_root,
    label,
    output,
    prompt,
    new_tokens,
    endpoint,
    qualification,
  ) = match args {
    [mode, contract, root, label, output, prompt, count] if mode == "prepare" ||
      mode == "simulate" ||
      mode == "simulate-fast" ||
      mode == "prepare-fast" =>
      (mode, contract, root, label, output, prompt, count, "", "")
    [
      mode,
      contract,
      root,
      label,
      output,
      prompt,
      count,
      endpoint,
      qualification,
    ] if mode == "physical" ||
      mode == "benchmark" ||
      mode == "benchmark-fast" ||
      mode == "benchmark-chain" ||
      mode == "benchmark-program-bursts" ||
      mode == "benchmark-stream-unary" ||
      mode == "benchmark-warm" ||
      mode == "generate-resident" ||
      mode == "generate-warm" =>
      (
        mode, contract, root, label, output, prompt, count, endpoint, qualification,
      )
    _ =>
      fail(
        "lunachip_qwen prepare|prepare-fast|simulate|simulate-fast|physical|benchmark|benchmark-fast|benchmark-chain|benchmark-program-bursts|benchmark-stream-unary|benchmark-warm|generate-resident|generate-warm CONTRACT MODEL_ROOT ASSOCIATION_LABEL_64_HEX NEW_OUTPUT PROMPT NEW_TOKENS [LOOPBACK_ENDPOINT QUALIFICATION_JSON [ROUTED_CLOCK_REPORT]]",
      )
  }
  let generate = mode == "generate-resident" || mode == "generate-warm"
  let resident_chain = generate ||
    mode == "benchmark-warm" ||
    mode == "benchmark-stream-unary" ||
    mode == "benchmark-program-bursts" ||
    mode == "benchmark-chain"
  let generation = @string.parse_int(new_tokens)
  guard generation >= 1 &&
    generation <= 16 &&
    output.has_prefix("/") &&
    !@fs.exists(output) else {
    fail("fresh absolute output and 1..16 generated tokens required")
  }
  let device = @device.parse_contract(@fs.read_file(contract).text())
  let root = @approved.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let config_file = root.open_file(
    @approved.ApprovedRelativeLocator::new("config.json"),
  )
  let config_bytes = config_file.read_immutable_snapshot(maximum_bytes=65536L)
  config_file.close()
  let metadata = @config.parse_qwen3_bytes(
    config_bytes[:],
    @spec.ContentDigest::from_sha256(label),
    max_bytes=65536,
    max_depth=16,
  )
  let tokenizer_file = root.open_file(
    @approved.ApprovedRelativeLocator::new("tokenizer.json"),
  )
  let tokenizer_bytes = tokenizer_file.read_immutable_snapshot(
    maximum_bytes=33554432L,
  )
  tokenizer_file.close()
  let tokenizer = @tokenizer_json.parse_bytes(
    tokenizer_bytes[:],
    digest=@tokenizer.TokenizerDigest::parse(
      @crypto.bytes_to_hex_string(@crypto.sha256(tokenizer_bytes)),
    ),
    limits={
      max_input_bytes: 4096,
      max_output_tokens: 64,
      max_decoded_bytes: 8192,
      max_vocab_entries: 160000,
      max_merge_rules: 160000,
      max_token_bytes: 1024,
      max_special_tokens: 256,
    },
    max_bytes=33554432,
    max_depth=24,
  )
  let encoded = tokenizer.encode_text(
    prompt,
    special_policy=Allow,
    output_limit=64,
    truncation=RejectOverflow,
  )
  let tokens = encoded.token_ids
  guard !tokens.is_empty() && tokens.length() + generation <= 64 else {
    fail("diagnostic limit is 64 total tokens")
  }
  let context = tokens.length() + generation
  let layout = @layout.prepare(
    device,
    metadata,
    context~,
    align_matrices_1k=@env.get_env_var("PBLAZE3_ALIGN_MATRIX_1K") == Some("1"),
  )
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=1048576,
    max_tensor_count=512,
    max_rank=4,
    max_name_length=256,
    max_payload_bytes=2147483648L,
    max_materialized_tensor_bytes=2147483648L,
  )
  let locator = @approved.ApprovedRelativeLocator::new("model.safetensors")
  let inspected = @weights.inspect_source_safetensors_file(
    root,
    metadata,
    locator,
    expected_artifact_sha256=label,
    limits=@weights.Qwen3SourceFileLimits::new(
      max_file_bytes=2147483648L,
      compare_chunk_bytes=1048576,
      safetensors=limits,
    ),
  )
  let file = root.open_file(locator)
  defer file.close()
  let stamp = file.stamp()
  let prefix = read_range(file, 0L, 8)
  let payload = @reader.declared_header_bytes(prefix[:], limits~).to_int64() +
    8L
  let source = Source::{
    file,
    payload,
    transfers: @qwen.prepare_transfers(layout, inspected.bindings()),
    layout,
  }
  @fs.mkdir(output)
  let weight_manifest : Array[Json] = []
  for t in source.transfers {
    weight_manifest.push({
      "ordinal": t.ordinal,
      "source_file_offset": payload + t.source_offset,
      "source_bytes": t.source_bytes,
      "ddr_address": t.destination.address,
    })
    if t.expanded_norm is Some(region) {
      let raw = source.weight(t.ordinal)
      let expanded = Array::make(raw.length() * 2, b'\x00')
      @qwen.expand_bf16(raw[:], expanded)
      @fs.write_file(
        output + "/norm-\{t.ordinal}.bin",
        Bytes::from_array(expanded[:]),
        create_mode=CreateNew,
      )
      weight_manifest.push({
        "file": "norm-\{t.ordinal}.bin",
        "ddr_address": region.address,
        "bytes": region.bytes,
      })
    }
  }
  let plans : Array[@qwen.DecodeStep] = []
  let manifests : Array[Json] = []
  for position in 0..<context {
    let plan = @qwen.prepare_decode_step(
      device,
      layout,
      position,
      resident=mode.has_suffix("-fast") || resident_chain,
      indirect_bf16=(
          resident_chain || mode == "simulate-fast" || mode == "prepare-fast"
        ) &&
        @env.get_env_var("PBLAZE3_BF16_INPUT") == Some("1"),
    )
    plans.push(plan)
    let files : Array[Json] = []
    for layer in 0..<plan.program_count() {
      let program = plan.program(layer).unwrap()
      for chunk in 0..<program.chunk_count() {
        let name = "p\{position}-l\{layer}-c\{chunk}.bin"
        @fs.write_file(
          output + "/" + name,
          program.chunk(chunk).unwrap(),
          create_mode=CreateNew,
        )
        files.push({ "layer": layer, "chunk": chunk, "file": name })
      }
    }
    let constants = "p\{position}-constants.bin"
    @fs.write_file(
      output + "/" + constants,
      plan.constants(),
      create_mode=CreateNew,
    )
    manifests.push({
      "position": position,
      "programs": files,
      "constants_file": constants,
      "constants_address": plan.constants_address(),
      "embedding_address": plan.embedding_address(),
      "logits_address": plan.logits_address(),
    })
  }
  let manifest : Json = {
    "schema": "lunaflux.pblaze3-full-qwen-diagnostic/v1",
    "mode": mode,
    "association_label": label,
    "checkpoint_payload_hashed": false,
    "identity_kind": "supplied-association-label",
    "source_file_bytes": stamp.byte_count(),
    "source_manifest_sha256": inspected.source_manifest_sha256(),
    "weights": weight_manifest,
    "steps": manifests,
    "context": context,
    "prompt": prompt,
    "prompt_tokens": tokens.copy(),
    "new_tokens": generation,
    "control_base": "0x80100000",
    "physical_execution": false,
  }
  @fs.write_file(
    output + "/manifest.json",
    manifest.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(
    "Prepared \{context} complete positions, \{source.transfers.length()} tensors, prompt tokens \{tokens}",
  )
  if mode == "simulate" || mode == "simulate-fast" {
    simulate(source, plans, tokenizer, tokens, generation, output)
  }
  if mode == "physical" {
    physical(
      source, plans, tokenizer, tokens, generation, output, endpoint, qualification,
    )
  }
  if mode == "benchmark" || mode == "benchmark-fast" || resident_chain {
    benchmark(
      source,
      plans,
      tokenizer,
      tokens,
      generation,
      output,
      endpoint,
      qualification,
      clock_report~,
      loader_ab=mode == "benchmark-program-bursts",
      unary_ab=mode == "benchmark-stream-unary",
      generate~,
      require_resident_weights=mode == "generate-warm" ||
        mode == "benchmark-warm",
      fast=mode == "benchmark-fast" || resident_chain,
      chain=resident_chain,
    )
  }
  assert_true(
    file.stamp() == stamp,
    msg="checkpoint changed during preparation/execution",
  )
}

///|
async fn simulate(
  source : Source,
  plans : Array[@qwen.DecodeStep],
  tokenizer : @tokenizer.TokenizerSpec,
  tokens : Array[Int32],
  generation : Int32,
  output : String,
) -> Unit {
  let machine = Emulator::new(source)
  let reference = Reference::new(source)
  let prompt_count = tokens.length()
  let results : Array[Json] = []
  for position in 0..<(prompt_count + generation - 1) {
    let token = tokens[position]
    let plan = plans[position]
    machine.write(plan.embedding_address(), source.step_input(plan, token))
    machine.write(plan.constants_address(), plan.constants())
    for layer in 0..<plan.program_count() {
      machine.execute(plan.program(layer).unwrap())
    }
    let actual = Array::makei(151936, i => {
      machine.read(plan.logits_address() + i.to_uint64() * 4UL)
    })
    let expected = reference.step(token, position)
    let observed_token = greedy(actual[:])
    let expected_token = greedy(expected[:])
    let mut max_error = 0.0
    let mut sum_square = 0.0
    for i in 0..<151936 {
      let difference = (actual[i].to_double() - expected[i].to_double()).abs()
      max_error = max_error.max(difference)
      sum_square += difference * difference
    }
    @fs.write_file(
      output + "/p\{position}-emulator-logits.bin",
      float_bytes(actual[:]),
      create_mode=CreateNew,
    )
    @fs.write_file(
      output + "/p\{position}-reference-logits.bin",
      float_bytes(expected[:]),
      create_mode=CreateNew,
    )
    results.push({
      "position": position,
      "input_token": token,
      "emulator_argmax": observed_token,
      "reference_argmax": expected_token,
      "max_absolute_logit_error": max_error,
      "rms_logit_error": (sum_square / 151936.0).sqrt(),
    })
    @fs.write_file(
      output + "/simulation-progress.json",
      Json::array(results.copy()).stringify(indent=2),
    )
    println(
      "position \{position}: device-ISA simulation token=\{observed_token}, reference=\{expected_token}, max logit error=\{max_error}",
    )
    assert_eq(observed_token, expected_token, msg="greedy divergence")
    assert_true(
      max_error <= 0.05,
      msg="approximation error exceeds diagnostic gate",
    )
    if position >= prompt_count - 1 {
      tokens.push(observed_token)
      if terminal_token(observed_token) {
        break
      }
    }
  }
  let result : Json = {
    "schema": "lunaflux.pblaze3-full-qwen-simulation/v1",
    "physical_execution": false,
    "compared_vocabulary": 151936,
    "positions": results,
    "tokens": tokens,
    "generated_text": tokenizer.decode_text(
      tokens[prompt_count:],
      special_policy=Preserve,
    ),
    "reference": "independent MoonBit Qwen equations, FP32 ordered reductions and libm nonlinearities",
    "gate": "all logits finite, max_abs<=0.05, exact greedy token at every position",
  }
  @fs.write_file(
    output + "/simulation-result.json",
    result.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(result.stringify(indent=2))
}
