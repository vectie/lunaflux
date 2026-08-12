///|
fn valid_document() -> String {
  (
    #|{
    #|  "schema_version": "lunaflux.config.v1alpha1",
    #|  "service": {
    #|    "listen_host": "127.0.0.1",
    #|    "listen_port": 8080,
    #|    "max_request_bytes": 1048576,
    #|    "graceful_drain_milliseconds": 30000,
    #|    "diagnostic_mode": "verbose"
    #|  },
    #|  "model": {
    #|    "artifact_path": "/models/reference",
    #|    "expected_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    #|    "architecture_allowlist": ["LlamaForCausalLM", "DenseDecoder"]
    #|  },
    #|  "device": {
    #|    "visible_devices": [3, 1],
    #|    "memory_ceiling_bytes": 8589934592
    #|  }
    #|}
  )
}

///|
test "string and bytes parsing construct deterministic focused records" {
  let text = valid_document()
  let from_string = @document.parse_string(text, max_bytes=4096)
  let from_bytes = @document.parse_bytes(@utf8.encode(text), max_bytes=4096)
  assert_eq(from_string.service(), from_bytes.service())
  assert_eq(from_string.model(), from_bytes.model())
  assert_eq(from_string.device(), from_bytes.device())
  assert_eq(from_string.resolved_plan(), from_bytes.resolved_plan())
  assert_eq(from_string.schema().version(), V1Alpha1)
  assert_eq(from_string.service().listen_port(), 8080)
  assert_eq(from_string.service().diagnostic_mode(), Verbose)
  assert_eq(from_string.model().architecture_allowlist()[0], "LlamaForCausalLM")
  assert_eq(from_string.device().visible_devices()[0], 3)
  assert_eq(from_string.device().memory_ceiling_bytes(), 8589934592L)
  assert_eq(from_string.resolved_plan().primary_device(), 3)
  assert_eq(
    from_string.resolved_plan().selected_architecture(),
    "LlamaForCausalLM",
  )
}

///|
test "malformed JSON and non-object roots fail closed" {
  try @document.parse_string("{", max_bytes=32) catch {
    error => debug_inspect(error, content="MalformedJson")
  } noraise {
    _ => fail("expected malformed JSON rejection")
  }
  try @document.parse_string("[]", max_bytes=32) catch {
    error => debug_inspect(error, content="NonObjectRoot")
  } noraise {
    _ => fail("expected non-object root rejection")
  }
}

///|
test "nested and escaped-equivalent duplicate fields fail before core parsing" {
  let nested = valid_document().replace(
    old="\"visible_devices\": [3, 1]",
    new="\"visible_devices\": [3, 1], \"visible_devices\": [0]",
  )
  try @document.parse_string(nested, max_bytes=4096) catch {
    error => debug_inspect(error, content="DuplicateField(depth=2, index=1)")
  } noraise {
    _ => fail("expected nested duplicate rejection")
  }
  let escaped = valid_document().replace(
    old="\"schema_version\": \"lunaflux.config.v1alpha1\",",
    new=(
      #|"schema_version": "lunaflux.config.v1alpha1",
      #|  "schema_\u0076ersion": "lunaflux.config.v1alpha1",
    ),
  )
  try @document.parse_string(escaped, max_bytes=4096) catch {
    error => debug_inspect(error, content="DuplicateField(depth=1, index=1)")
  } noraise {
    _ => fail("expected escaped duplicate rejection")
  }
}

///|
test "unknown and missing fields fail without echoing their payload" {
  let unknown = valid_document().replace(
    old="\"listen_port\": 8080,",
    new="\"listen_port\": 8080, \"secret_payload\": \"do-not-echo\",",
  )
  try @document.parse_string(unknown, max_bytes=4096) catch {
    error =>
      debug_inspect(error, content="UnknownField(area=ServiceArea, index=2)")
  } noraise {
    _ => fail("expected unknown-field rejection")
  }
  let missing = valid_document().replace(
    old=(
      #|    "graceful_drain_milliseconds": 30000,
      #|    "diagnostic_mode": "verbose"
    ),
    new=(
      #|    "graceful_drain_milliseconds": 30000
    ),
  )
  try @document.parse_string(missing, max_bytes=4096) catch {
    error =>
      debug_inspect(
        error,
        content="MissingField(area=ServiceArea, field=DiagnosticModeField)",
      )
  } noraise {
    _ => fail("expected missing-field rejection")
  }
}

///|
test "wrong types and non-integral numbers are rejected" {
  let wrong_type = valid_document().replace(
    old="\"listen_port\": 8080",
    new="\"listen_port\": \"8080\"",
  )
  try @document.parse_string(wrong_type, max_bytes=4096) catch {
    error =>
      debug_inspect(
        error,
        content=(
          #|InvalidValue(
          #|  area=ServiceArea,
          #|  field=ListenPortField,
          #|  issue=ExpectedInteger,
          #|  index=0,
          #|)
        ),
      )
  } noraise {
    _ => fail("expected type rejection")
  }
  let fractional = valid_document().replace(
    old="\"listen_port\": 8080",
    new="\"listen_port\": 8080.5",
  )
  try @document.parse_string(fractional, max_bytes=4096) catch {
    error =>
      debug_inspect(
        error,
        content=(
          #|InvalidValue(
          #|  area=ServiceArea,
          #|  field=ListenPortField,
          #|  issue=ExpectedInteger,
          #|  index=0,
          #|)
        ),
      )
  } noraise {
    _ => fail("expected non-integral rejection")
  }
}

///|
test "focused integer ranges are enforced during construction" {
  let out_of_range = valid_document().replace(
    old="\"listen_port\": 8080",
    new="\"listen_port\": 65536",
  )
  try @document.parse_string(out_of_range, max_bytes=4096) catch {
    error =>
      debug_inspect(
        error,
        content=(
          #|InvalidValue(
          #|  area=ServiceArea,
          #|  field=ListenPortField,
          #|  issue=RejectedByFocusedValidator,
          #|  index=0,
          #|)
        ),
      )
  } noraise {
    _ => fail("expected range rejection")
  }
}

///|
test "byte limits and invalid UTF-8 are rejected before JSON parsing" {
  try @document.parse_string(valid_document(), max_bytes=16) catch {
    error => debug_inspect(error, content="ByteLimitExceeded(limit=16)")
  } noraise {
    _ => fail("expected oversized document rejection")
  }
  try @document.parse_bytes([0xff], max_bytes=1) catch {
    error => debug_inspect(error, content="InvalidUtf8")
  } noraise {
    _ => fail("expected UTF-8 rejection")
  }
  try @document.parse_bytes(b"{}", max_bytes=0) catch {
    error => debug_inspect(error, content="InvalidByteLimit")
  } noraise {
    _ => fail("expected invalid limit rejection")
  }
}
