///|
suberror OpenAIResponsesReplayCliError {
  InvalidArguments
  InvalidRoot
  ReadFailure
  ReplayRejected
  Phase4Rejected
} derive(Debug, Eq)

///|
fn capture_name(index : Int) -> String {
  let digits = index.to_string()
  let prefix = match digits.length() {
    1 => "00"
    2 => "0"
    _ => ""
  }
  "trial-" + prefix + digits + ".capture.json"
}

///|
async fn read_regular_capture(
  root : String,
  relative : String,
) -> Bytes raise OpenAIResponsesReplayCliError {
  let expected = root + "/" + relative
  let canonical = @fs.realpath(expected) catch { _ => raise ReadFailure }
  if canonical != expected {
    raise ReadFailure
  }
  @fs.read_file(canonical).binary() catch {
    _ => raise ReadFailure
  }
}

///|
async fn read_absolute_regular(
  supplied : String,
) -> Bytes raise OpenAIResponsesReplayCliError {
  if !supplied.has_prefix("/") {
    raise InvalidRoot
  }
  let canonical = @fs.realpath(supplied) catch { _ => raise ReadFailure }
  if canonical != supplied {
    raise ReadFailure
  }
  @fs.read_file(canonical).binary() catch {
    _ => raise ReadFailure
  }
}

///|
async fn read_phase4_pinned_policy(
  supplied : String,
) -> Bytes raise OpenAIResponsesReplayCliError {
  if supplied == "/dev/fd/8" {
    return @fs.read_file(supplied).binary() catch { _ => raise ReadFailure }
  }
  read_absolute_regular(supplied)
}

///|
async fn run_preflight(
  arguments : ArrayView[String],
) -> Unit raise OpenAIResponsesReplayCliError {
  guard arguments is [_, "--preflight", declaration_path, expected_sha256] else {
    raise InvalidArguments
  }
  let declaration = @openai_campaign.OpenAIResponsesCampaignDeclaration::parse(
    read_absolute_regular(declaration_path),
    expected_sha256,
  ) catch {
    _ => raise ReplayRejected
  }
  println("schema=lunaflux-openai-responses-campaign-preflight.v1")
  println("declaration_sha256=" + declaration.digest().as_hex())
  println("engine_order=lunaflux,vllm,sglang")
  println(
    "profile_order=latency,chat,long_prefill,decode_heavy,prefix_rich," +
    "prefix_cold,saturation,churn,mixed",
  )
  println("trial_order=profile_then_ordinal_then_counterbalanced_engine")
  println("trial_count=81")
  println("credential_ingress=external.inherited-descriptor.v1")
  let engine_names = ["lunaflux", "vllm", "sglang"]
  for index, engine in engine_names {
    let revision = declaration.engine_revision(index) catch {
      _ => raise ReplayRejected
    }
    let image = declaration.engine_image(index) catch {
      _ => raise ReplayRejected
    }
    let configuration = declaration.engine_configuration(index) catch {
      _ => raise ReplayRejected
    }
    let executable = declaration.engine_executable(index) catch {
      _ => raise ReplayRejected
    }
    let endpoint = declaration.endpoint(index) catch {
      _ => raise ReplayRejected
    }
    println("engine_" + index.to_string() + "=" + engine)
    println(
      "engine_" + index.to_string() + "_revision_sha256=" + revision.as_hex(),
    )
    println("engine_" + index.to_string() + "_image_sha256=" + image.as_hex())
    println(
      "engine_" +
      index.to_string() +
      "_configuration_sha256=" +
      configuration.as_hex(),
    )
    println(
      "engine_" +
      index.to_string() +
      "_executable_sha256=" +
      executable.as_hex(),
    )
    println("endpoint_" + index.to_string() + "=" + endpoint)
  }
  println("comparison_authority=none")
}

///|
async fn run_replay(
  arguments : ArrayView[String],
) -> Unit raise OpenAIResponsesReplayCliError {
  guard arguments is [_, supplied_root, expected_declaration_sha256] else {
    raise InvalidArguments
  }
  let root = @fs.realpath(supplied_root) catch { _ => raise InvalidRoot }
  if root != supplied_root || !root.has_prefix("/") {
    raise InvalidRoot
  }
  let declaration_bytes = read_regular_capture(
    root, "campaign.declaration.json",
  )
  if declaration_bytes.length() <= 0 ||
    declaration_bytes.length() > PHASE4_MAX_DECLARATION_BYTES {
    raise ReplayRejected
  }
  let declaration = @openai_campaign.OpenAIResponsesCampaignDeclaration::parse(
    declaration_bytes, expected_declaration_sha256,
  ) catch {
    _ => raise ReplayRejected
  }
  let captures : Array[Bytes] = []
  for index in 0..<81 {
    captures.push(read_regular_capture(root, capture_name(index)))
  }
  let evidence = @openai_campaign.replay_openai_responses_campaign(
    declaration, captures,
  ) catch {
    _ => raise ReplayRejected
  }
  println(
    @utf8.decode(evidence.canonical_bytes()) catch {
      _ => raise ReplayRejected
    },
  )
  println("replay_record_sha256=" + evidence.digest().as_hex())
}

///|
async fn run_phase4_prefix_gate(
  arguments : ArrayView[String],
) -> Unit raise OpenAIResponsesReplayCliError {
  guard arguments
    is [
      _,
      "--phase4-prefix-gate",
      supplied_root,
      policy_path,
      expected_policy_sha256,
      expected_gate_executable_sha256,
    ] else {
    raise InvalidArguments
  }
  let root = @fs.realpath(supplied_root) catch { _ => raise InvalidRoot }
  if root != supplied_root || !root.has_prefix("/") {
    raise InvalidRoot
  }
  let policy = phase4_parse_policy(
    read_phase4_pinned_policy(policy_path),
    expected_policy_sha256,
  ) catch {
    _ => raise Phase4Rejected
  }
  if policy.gate_executable != expected_gate_executable_sha256 {
    raise Phase4Rejected
  }
  phase4_join_handoff(root, policy) catch {
    _ => raise Phase4Rejected
  }
  let declaration_bytes = read_regular_capture(
    root, "campaign.declaration.json",
  )
  if declaration_bytes.length() <= 0 ||
    declaration_bytes.length() > PHASE4_MAX_DECLARATION_BYTES {
    raise Phase4Rejected
  }
  let declaration = @openai_campaign.OpenAIResponsesCampaignDeclaration::parse(
    declaration_bytes,
    policy.declaration,
  ) catch {
    _ => raise ReplayRejected
  }
  phase4_join_declaration(policy, declaration_bytes) catch {
    _ => raise Phase4Rejected
  }
  let captures : Array[Bytes] = []
  let mut capture_set_bytes = 0L
  for index in 0..<PHASE4_CORRECTNESS_TRIALS {
    let capture = read_regular_capture(root, capture_name(index))
    if capture.length() <= 0 || capture.length() > PHASE4_MAX_CAPTURE_BYTES {
      raise Phase4Rejected
    }
    capture_set_bytes = capture_set_bytes + Int64::from_int(capture.length())
    if capture_set_bytes > PHASE4_MAX_CAPTURE_SET_BYTES {
      raise Phase4Rejected
    }
    captures.push(capture)
  }
  let replay = @openai_campaign.replay_openai_responses_campaign(
    declaration, captures,
  ) catch {
    _ => raise ReplayRejected
  }
  let correctness = phase4_load_correctness(
    root,
    policy.declaration,
    captures,
    policy.correctness_inventory,
  ) catch {
    _ => raise Phase4Rejected
  }
  let evidence = phase4_gate_evidence(
    replay.measurements(),
    correctness,
    policy,
  ) catch {
    _ => raise Phase4Rejected
  }
  println(
    @utf8.decode(evidence.canonical_bytes()) catch {
      _ => raise Phase4Rejected
    },
  )
  println("phase4_prefix_gate_sha256=" + evidence.digest().as_hex())
  if !evidence.passed() {
    raise Phase4Rejected
  }
}

///|
async fn main {
  let arguments = @env.args()
  if arguments.length() > 1 && arguments[1] == "--preflight" {
    try! run_preflight(arguments)
  } else if arguments.length() > 1 && arguments[1] == "--phase4-prefix-gate" {
    try! run_phase4_prefix_gate(arguments)
  } else {
    try! run_replay(arguments)
  }
}
