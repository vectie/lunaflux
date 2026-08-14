///|
fn print_usage() -> Unit {
  println("LunaFlux \{@contracts.protocol_version()}")
  println("usage:")
  println("  lunaflux doctor")
  println("  lunaflux plan MODEL_PATH sha256:DIGEST")
  println(
    "  lunaflux reference ROOT CONFIG_REL TOKENIZER_REL WEIGHTS_REL CONFIG_SHA TOKENIZER_SHA WEIGHTS_SHA MAX_NEW TOKEN_IDS",
  )
  println("  lunaflux version")
}

///|
const REFERENCE_MAX_TOKENS : Int = 4096

///|
enum ReferenceArgument {
  MaximumNewTokens
  TokenIds
} derive(Debug, Eq)

///|
suberror CliError {
  InvalidReferenceArgument(ReferenceArgument)
} derive(Debug, Eq)

///|
fn parse_positive(text : StringView) -> Int raise CliError {
  let value : Int = @strconv.from_str(text) catch {
    _ => raise InvalidReferenceArgument(MaximumNewTokens)
  }
  if value <= 0 || value > REFERENCE_MAX_TOKENS {
    raise InvalidReferenceArgument(MaximumNewTokens)
  }
  value
}

///|
fn parse_token_ids(text : StringView) -> Array[Int] raise CliError {
  let tokens : Array[Int] = []
  for part in text.split(",") {
    if part.is_empty() || tokens.length() >= REFERENCE_MAX_TOKENS {
      raise InvalidReferenceArgument(TokenIds)
    }
    let token : Int = @strconv.from_str(part) catch {
      _ => raise InvalidReferenceArgument(TokenIds)
    }
    if token < 0 {
      raise InvalidReferenceArgument(TokenIds)
    }
    tokens.push(token)
  }
  if tokens.is_empty() {
    raise InvalidReferenceArgument(TokenIds)
  }
  tokens
}

///|
fn reference_artifact_limits() -> @artifact.ArtifactLimits raise {
  @artifact.ArtifactLimits::new(
    max_model_config_bytes=1048576L,
    max_tokenizer_bytes=67108864L,
    max_weights_bytes=67108864L,
    max_total_bytes=135266304L,
    max_model_config_depth=32,
    max_tokenizer_depth=64,
    tokenizer_limits={
      max_input_bytes: 1048576,
      max_output_tokens: REFERENCE_MAX_TOKENS,
      max_decoded_bytes: 4194304,
      max_vocab_entries: 1048576,
      max_merge_rules: 1048576,
      max_token_bytes: 4096,
      max_special_tokens: 4096,
    },
    safetensors_limits=@safetensors.HeaderLimits::new(
      max_header_bytes=67108864,
      max_tensor_count=4096,
      max_rank=2,
      max_name_length=512,
      max_payload_bytes=67108864L,
      max_materialized_tensor_bytes=67108864L,
    ),
  )
}

///|
fn reference_execution_limits() -> @reference.ExecutionLimits raise {
  @reference.ExecutionLimits::new(
    kernels=@refkernels.ReferenceLimits::new(
      max_rows=REFERENCE_MAX_TOKENS,
      max_columns=1048576,
      max_elements=33554432,
    ),
    sampling=@sampling.SamplingLimits::new(1, 1048576),
    max_operations=8192,
    max_tensors=4096,
    max_input_tokens=REFERENCE_MAX_TOKENS,
  )
}

///|
fn print_tokens(tokens : ReadOnlyArray[Int]) -> Unit {
  let output = StringBuilder()
  for index, token in tokens {
    if index > 0 {
      output.write_string(",")
    }
    output.write_string(token.to_string())
  }
  println("generated token IDs: \{output}")
}

///|
fn run_reference(
  root_path : String,
  config_locator : String,
  tokenizer_locator : String,
  weights_locator : String,
  config_sha : String,
  tokenizer_sha : String,
  weights_sha : String,
  maximum_new_text : String,
  token_text : String,
) -> Unit raise {
  let maximum_new = parse_positive(maximum_new_text)
  let prompt = parse_token_ids(token_text)
  if maximum_new > REFERENCE_MAX_TOKENS - prompt.length() {
    raise InvalidReferenceArgument(MaximumNewTokens)
  }
  let source = @artifact.ArtifactSource::new(
    model_config=config_locator,
    tokenizer=tokenizer_locator,
    weights=weights_locator,
  )
  let expected = @artifact.ExpectedDigests::new(
    model_config=@artifact.ArtifactDigest::parse(config_sha),
    tokenizer=@artifact.ArtifactDigest::parse(tokenizer_sha),
    weights=@artifact.ArtifactDigest::parse(weights_sha),
  )
  let artifact_limits = reference_artifact_limits()
  // Absolute path admission is confined to this deployment boundary. The
  // artifact package sees only the pinned root and opaque relative source.
  let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
  let bundle = @artifact.load_reference_bundle(
    root, source, expected, artifact_limits,
  ) catch {
    error => {
      root.close() catch {
        _ => ()
      }
      raise error
    }
  }
  // No execution result or console output is published until the caller-owned
  // root has been deterministically consumed.
  root.close()
  let admitted = @llama_admission.complete(bundle.model(), bundle.safetensors())
  let host = @materialize.copy_to_host(
    admitted.bindings(),
    bundle.weights_bytes(),
    limits=@materialize.MaterializationLimits::new(
      max_header_bytes=67108864,
      max_tensor_count=4096,
      max_payload_bytes=67108864L,
      max_host_bytes=67108864L,
    ),
  )
  let limits = reference_execution_limits()
  let prepared = @reference.prepare(admitted.plan(), host, limits~)
  let generated = @reference.generate(
    admitted.plan(),
    prepared,
    prompt,
    max_new_tokens=maximum_new,
    limits~,
  )
  println("LunaFlux offline reference execution")
  println("model content sha256: \{generated.identity().content().as_hex()}")
  println("input token count: \{generated.input_tokens()}")
  print_tokens(generated.generated_tokens())
  println("production readiness: false; correctness path only")
}

///|
fn unavailable_device_reason(reason : @hardware.UnavailableReason) -> String {
  match reason {
    UnsupportedPlatform => "unsupported platform"
    DriverLibraryMissing => "driver library missing"
    DriverAbiIncomplete => "driver ABI incomplete"
    DriverInitializationFailed => "driver initialization failed"
    NoDevices => "no devices"
    InventoryFailed => "inventory failed"
  }
}

///|
fn print_doctor() -> Unit {
  println("LunaFlux foundation inspection")
  println("contracts: ready")
  println("configuration validators: ready")
  println("bounded tokenizer JSON admission: ready")
  println("bounded model metadata and weight-manifest validation: ready")
  match @hardware.probe() {
    Available(driver_version~, devices~) =>
      println(
        "CUDA device probe: available; driver \{driver_version}; devices \{devices.length()}",
      )
    Unavailable(reason) =>
      println(
        "CUDA device probe: unavailable; \{unavailable_device_reason(reason)}",
      )
  }
  println("offline reference executor: ready (correctness only)")
  println("bounded reference bundle loading: ready (whole-file snapshot)")
  println(
    "streaming device weight loading: implemented (approved read-only regular-file contract)",
  )
  println("exact static device executor: implemented")
  println(
    "physical CUDA correctness and resource gates: not proven on this host",
  )
  println("readiness: false")
}

///|
fn print_plan(model_path : String, digest : String) -> Unit raise {
  let service = @service.new("127.0.0.1", 8080, 1048576, 30000, Normal)
  let model = @model.new(model_path, digest, ["LlamaForCausalLM"])
  let device = @device.new([0], 8589934592L)
  let plan = @resolved.resolve(service, model, device)
  println("LunaFlux configuration-only startup plan")
  println("artifact identity: validated; path redacted")
  println("architecture: \{plan.selected_architecture()}")
  println("primary device: \{plan.primary_device()} (configured, not probed)")
  println("memory ceiling bytes: \{plan.memory_ceiling_bytes()}")
  println("decisions recorded: \{plan.decisions().length()}")
  println("model materialization: not performed")
  println("readiness: false; CUDA capability resolution is a later gate")
}

///|
fn main raise {
  match @env.args() {
    [_, "doctor"] => print_doctor()
    [_, "plan", model_path, digest] => print_plan(model_path, digest)
    [
      _,
      "reference",
      root_path,
      config_locator,
      tokenizer_locator,
      weights_locator,
      config_sha,
      tokenizer_sha,
      weights_sha,
      maximum_new,
      token_ids,
    ] =>
      run_reference(
        root_path, config_locator, tokenizer_locator, weights_locator, config_sha,
        tokenizer_sha, weights_sha, maximum_new, token_ids,
      )
    [_, "version"] => println(@contracts.protocol_version())
    _ => print_usage()
  }
}
