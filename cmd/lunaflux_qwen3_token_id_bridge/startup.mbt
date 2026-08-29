///|
fn bridge_is_sha256(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
fn bridge_parse_positive(
  value : String,
  label : String,
) -> Int raise BridgeRuntimeError {
  let parsed = @string.parse_int(value, base=10) catch {
    _ => raise InvalidArgument(label + " is not numeric")
  }
  if parsed <= 0 {
    raise InvalidArgument(label + " is not positive")
  }
  parsed
}

///|
fn bridge_digest(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn bridge_identity_path(
  argument : String,
  label : String,
) -> String raise BridgeRuntimeError {
  let marker = "#sha256="
  guard argument.find(marker) is Some(index) && index > 0 else {
    raise InvalidArgument(label + " is not digest suffixed")
  }
  argument[:index].to_owned()
}

///|
async fn bridge_authenticate_file(
  argument : String,
  maximum_bytes : Int,
  label : String,
) -> Unit raise BridgeRuntimeError {
  let marker = "#sha256="
  let path = bridge_identity_path(argument, label)
  let index = path.length()
  let expected = argument[index + marker.length():].to_owned()
  if !path.has_prefix("/") || !bridge_is_sha256(expected) {
    raise InvalidArgument(label + " identity is malformed")
  }
  let canonical = @fs.realpath(path) catch {
    _ => raise StartupIdentity(label + " path is unavailable")
  }
  if canonical != path {
    raise StartupIdentity(label + " path is not canonical")
  }
  let bytes = @fs.read_file(path).binary() catch {
    _ => raise StartupIdentity(label + " is unreadable")
  }
  if bytes.length() <= 0 ||
    bytes.length() > maximum_bytes ||
    bridge_digest(bytes) != expected {
    raise StartupIdentity(label + " digest or size differs")
  }
}

///|
async fn bridge_config(
  args : Array[String],
) -> BridgeConfig raise BridgeRuntimeError {
  if args.length() != 13 {
    raise InvalidArgument(
      "usage: lunaflux_qwen3_token_id_bridge BRIDGE_EXECUTABLE#sha256=HEX TOKENIZER_JSON#sha256=HEX LAUNCH#sha256=HEX C32_CAPACITY_RECEIPT#sha256=HEX LISTEN_ADDR RUNTIME_ADDR MODEL_CONTENT_SHA256 MODEL_PLAN_SHA256 MAX_INPUT_TOKENS MAX_OUTPUT_TOKENS MAX_TOKEN_ID MAX_CONTEXT_TOKENS",
    )
  }
  bridge_authenticate_file(args[1], 134217728, "bridge executable")
  bridge_authenticate_file(args[2], 33554432, "Qwen tokenizer")
  bridge_authenticate_file(args[3], 16777216, "Qwen launch")
  bridge_authenticate_file(args[4], 1048576, "concurrency-32 capacity receipt")
  let actual_executable = @fs.realpath(args[0]) catch {
    _ => raise StartupIdentity("running bridge executable is unavailable")
  }
  if actual_executable != bridge_identity_path(args[1], "bridge executable") {
    raise StartupIdentity("running bridge executable identity differs")
  }
  if !bridge_is_sha256(args[7]) || !bridge_is_sha256(args[8]) {
    raise InvalidArgument("model identity is malformed")
  }
  let max_input_tokens = bridge_parse_positive(args[9], "input token limit")
  let max_output_tokens = bridge_parse_positive(args[10], "output token limit")
  let max_token_id = bridge_parse_positive(args[11], "maximum token ID")
  let max_context_tokens = bridge_parse_positive(args[12], "context limit")
  if max_input_tokens + max_output_tokens > max_context_tokens {
    raise InvalidArgument("bridge token limits exceed the context limit")
  }
  if !args[5].has_prefix("127.0.0.1:") || !args[6].has_prefix("127.0.0.1:") {
    raise InvalidArgument("bridge and native service must use exact loopback")
  }
  {
    listen_addr: @socket.Addr::parse(args[5]) catch {
      _ => raise InvalidArgument("listen address is malformed")
    },
    runtime_addr: @socket.Addr::parse(args[6]) catch {
      _ => raise InvalidArgument("native runtime address is malformed")
    },
    model_content_sha256: args[7],
    model_plan_sha256: args[8],
    max_input_tokens,
    max_output_tokens,
    max_token_id,
    max_context_tokens,
  }
}
