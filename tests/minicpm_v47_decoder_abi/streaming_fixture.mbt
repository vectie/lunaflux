///|
/// Client-side fixture reader; transport/output arithmetic remains production
/// code. Return the assembled visible content and original terminal record.
async fn read_chat_stream(client : @http.Client) -> (String, Json) {
  let mut content = ""
  let mut terminal : Json? = None
  let mut done = false
  while client.read_until("\n\n") is Some(frame) {
    guard frame.trim().strip_prefix("data: ") is Some(payload) && !done else {
      fail("invalid or post-terminal SSE frame")
    }
    if payload == "[DONE]" {
      done = true
      continue
    }
    let chunk = @json.parse(payload)
    guard chunk is { "object": "chat.completion.chunk", .. } else {
      fail("invalid streaming chat object")
    }
    match chunk {
      { "choices": [{ "delta": { "content": String(text), .. }, .. }], .. } =>
        content += text
      { "choices": [{ "delta": { "role": "assistant", .. }, .. }], .. } => ()
      _ => fail("unexpected streaming chat chunk")
    }
    if chunk
      is { "choices": [{ "finish_reason": String(_), .. }], "usage": _, .. } {
      guard terminal is None else { fail("duplicate streaming terminal event") }
      terminal = Some(chunk)
    }
  }
  guard done && terminal is Some(record) else {
    fail("stream closed without terminal usage and DONE")
  }
  (content, record)
}
