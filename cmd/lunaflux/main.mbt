///|
fn print_usage() -> Unit {
  println("LunaFlux \{@contracts.protocol_version()}")
  println("usage:")
  println("  lunaflux run ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>")
  println(
    "  lunaflux validate-release ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>",
  )
  println(
    "  lunaflux validate-materialized-release ABSOLUTE_BUNDLE_ROOT#sha256=<64-lowercase-hex>",
  )
  println(
    "  lunaflux doctor ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>",
  )
  println("  lunaflux plan ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>")
  println("  lunaflux bench ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>")
  println(
    "  lunaflux inspect-kernels ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>",
  )
  println("legacy compatibility (never ready):")
  println("  lunaflux legacy-doctor")
  println("  lunaflux legacy-doctor --json")
  println("  lunaflux legacy-plan MODEL")
  println("  lunaflux legacy-inspect-kernels MODEL")
  println(
    "  lunaflux legacy-run MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256",
  )
  println(
    "  lunaflux legacy-doctor MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256",
  )
  println(
    "  lunaflux legacy-plan MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256",
  )
  println(
    "  lunaflux legacy-bench MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256",
  )
  println(
    "  lunaflux legacy-inspect-kernels MODEL_ROOT KERNEL_ROOT DESCRIPTOR_REL DESCRIPTOR_SHA256",
  )
  println(
    "  lunaflux legacy-config-plan ROOT CONFIG_REL CONFIG_SHA256 DECLARED_MODEL_SHA256 KV_BUDGET_BYTES TOKENS_PER_PAGE",
  )
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
  let bundle_result : Result[
    @artifact.ReferenceArtifactBundle,
    @artifact.ArtifactError,
  ] = Ok(
    @artifact.load_reference_bundle(root, source, expected, artifact_limits),
  ) catch {
    error => Err(error)
  }
  let bundle = match bundle_result {
    Ok(value) => value
    Err(error) => {
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
async fn run_cli(arguments : ArrayView[String]) -> Unit {
  match exact_opaque_run_argument(arguments) {
    Some(model) => run_native_model(model)
    None =>
      match exact_opaque_bench_argument(arguments) {
        Some(deployment) => run_native_benchmark(deployment)
        None =>
          match exact_release_diagnostic_argument(arguments) {
            Some(command) => run_release_diagnostic(command)
            None =>
              match arguments {
                [
                  _,
                  command,
                  model_root,
                  kernel_root,
                  descriptor_relative,
                  descriptor_sha256,
                ] =>
                  match legacy_pinned_command(command) {
                    Some(legacy_command) =>
                      run_pinned_operator_command(
                        legacy_command, model_root, kernel_root, descriptor_relative,
                        descriptor_sha256,
                      )
                    None => print_usage()
                  }
                [_, "validate-release", deployment] =>
                  validate_release(deployment)
                [_, "validate-materialized-release", bundle] =>
                  validate_materialized_release(bundle)
                [_, "legacy-doctor"] => print_doctor()
                [_, "legacy-doctor", "--json"] => print_doctor_json()
                [_, "legacy-plan", model_root] =>
                  run_operator_command("plan", model_root)
                [_, "legacy-inspect-kernels", model_root] =>
                  run_operator_command("inspect-kernels", model_root)
                [
                  _,
                  "legacy-config-plan",
                  root_path,
                  config_locator,
                  config_sha,
                  model_sha,
                  kv_budget_bytes,
                  tokens_per_page,
                ] =>
                  @lunaflux_plan.run(
                    root_path, config_locator, config_sha, model_sha, kv_budget_bytes,
                    tokens_per_page,
                  )
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
                    root_path, config_locator, tokenizer_locator, weights_locator,
                    config_sha, tokenizer_sha, weights_sha, maximum_new, token_ids,
                  )
                [_, "version"] => println(@contracts.protocol_version())
                _ => print_usage()
              }
          }
      }
  }
}

///|
async fn main {
  try! run_cli(@env.args())
}
