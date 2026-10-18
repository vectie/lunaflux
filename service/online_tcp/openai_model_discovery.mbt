///|
/// Construct once from the admitted deployment envelope, never from model
/// payloads or a device query. HTTP comparison/observation stays out of IR.
fn luna_openai_model_response(
  alias : FixedArray[Byte],
  offset : Int,
  length : Int,
  context : Int,
) -> Bytes raise LunaOnlineTcpError {
  if offset < 0 ||
    length <= 0 ||
    offset > alias.length() - length ||
    context <= 0 {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let encoded = Bytes::makei(length, i => alias[offset + i])
  let name = @utf8.decode(encoded) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let body : Json = {
    "object": "list",
    "data": [
      {
        "id": name,
        "object": "model",
        "owned_by": "lunaflux",
        "max_model_len": context,
      },
    ],
  }
  let text = body.stringify()
  let size = @utf8.encode(text).length()
  @utf8.encode(
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \{size}\r\nConnection: close\r\n\r\n" +
    text,
  )
}

///|
fn LunaOpenAIEmbeddedControl::prepare_models(self : Self) -> Bool {
  if self.models_response.length() == 0 {
    return self.prepare_static(LunaHttp1ControlBadRequest)
  }
  self.output = self.models_response
  self.output_cursor = 0
  self.output_length = self.output.length()
  true
}
