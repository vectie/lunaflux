///|
fn bridge_fixture(
  model? : String = "Qwen3-0.6B",
  sampling? : String = "{\"mode\":\"greedy\",\"temperature\":0,\"top_p\":1,\"seed\":0,\"ignore_eos\":true}",
  input? : String = "[151644,872]",
  output? : String = "32",
  stream? : String = "true",
) -> String {
  "{\"schema\":\"lunaflux.benchmark-token-ids-request.v1\",\"model\":\"" +
  model +
  "\",\"input_token_ids\":" +
  input +
  ",\"max_output_tokens\":" +
  output +
  ",\"sampling\":" +
  sampling +
  ",\"stream\":" +
  stream +
  "}"
}

///|
fn bridge_fixture_with_extra_field() -> String {
  let value = bridge_fixture()
  value[:value.length() - 1].to_owned() + ",\"extra\":1}"
}

///|
test "exact Qwen token-ID benchmark request is admitted" {
  let request = try! parse_qwen3_token_id_request(
    bridge_fixture(),
    4096,
    256,
    151935,
  )
  assert_eq(request.input_token_ids(), [151644, 872])
  assert_eq(request.max_output_tokens(), 32)
  let sse = Qwen3TokenSseBuffer::new()
  assert_eq(
    @utf8.decode(sse.stage(92648, b"Qwen")),
    "data: {\"schema\":\"lunaflux.benchmark-token.v1\",\"token_id\":92648,\"text\":\"Qwen\"}\n\n",
  )
  assert_eq(
    @utf8.decode(sse.stage(0, b"")),
    "data: {\"schema\":\"lunaflux.benchmark-token.v1\",\"token_id\":0,\"text\":\"\"}\n\n",
  )
  assert_eq(
    @utf8.decode(sse.terminal(b"")),
    "data: {\"schema\":\"lunaflux.benchmark-terminal.v1\",\"text\":\"\"}\n\ndata: [DONE]\n\n",
  )
}

///|
test "hostile Qwen token-ID benchmark requests fail closed" {
  for
    hostile in [
      bridge_fixture(model="foreign/model"),
      bridge_fixture(stream="false"),
      bridge_fixture(input="[]"),
      bridge_fixture(input="[151936]"),
      bridge_fixture(output="257"),
      bridge_fixture(
        sampling="{\"mode\":\"greedy\",\"temperature\":1,\"top_p\":1,\"seed\":0,\"ignore_eos\":true}",
      ),
      bridge_fixture(
        sampling="{\"mode\":\"greedy\",\"temperature\":0,\"top_p\":1,\"seed\":0,\"ignore_eos\":false}",
      ),
      bridge_fixture_with_extra_field(),
      "{\"schema\":\"lunaflux.benchmark-token-ids-request.v1\",\"schema\":\"lunaflux.benchmark-token-ids-request.v1\"}",
    ] {
    try parse_qwen3_token_id_request(hostile, 4096, 256, 151935) catch {
      MalformedRequest | UnsupportedRequest | RequestTooLarge => ()
    } noraise {
      _ => fail("hostile token-ID request was admitted")
    }
  }
}
