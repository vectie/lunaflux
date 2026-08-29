///|
fn luna_openai_responses_part(event_kind : Int, part : Int) -> Int {
  match event_kind {
    LUNA_OPENAI_EVENT_ACCEPTED =>
      match part {
        0 | 3 | 5 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_TOKEN =>
      match part {
        0 | 3 | 5 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_PAYLOAD
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_USAGE =>
      match part {
        0 | 3 | 5 | 7 | 9 | 11 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_INPUT
        6 => LUNA_OPENAI_PART_CACHED
        8 => LUNA_OPENAI_PART_OUTPUT
        10 => LUNA_OPENAI_PART_TOTAL
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_COMPLETED =>
      match part {
        0 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        6 => LUNA_OPENAI_PART_FINISH
        8 => LUNA_OPENAI_PART_PAYLOAD
        10 => LUNA_OPENAI_PART_INPUT
        12 => LUNA_OPENAI_PART_CACHED
        14 => LUNA_OPENAI_PART_OUTPUT
        16 => LUNA_OPENAI_PART_TOTAL
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_FAILED =>
      match part {
        0 | 3 | 5 | 7 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_PAYLOAD
        6 => LUNA_OPENAI_PART_RETRYABLE
        _ => LUNA_OPENAI_PART_END
      }
    _ => LUNA_OPENAI_PART_END
  }
}

///|
fn luna_openai_chat_part(event_kind : Int, part : Int) -> Int {
  match event_kind {
    LUNA_OPENAI_EVENT_ACCEPTED =>
      match part {
        0 | 3 | 5 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_TOKEN =>
      match part {
        0 | 3 | 5 | 7 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        6 => LUNA_OPENAI_PART_PAYLOAD
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_USAGE =>
      match part {
        0 | 3 | 5 | 7 | 9 | 11 | 13 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        6 => LUNA_OPENAI_PART_INPUT
        8 => LUNA_OPENAI_PART_CACHED
        10 => LUNA_OPENAI_PART_OUTPUT
        12 => LUNA_OPENAI_PART_TOTAL
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_COMPLETED =>
      match part {
        0 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        6 => LUNA_OPENAI_PART_PAYLOAD
        8 => LUNA_OPENAI_PART_FINISH
        10 => LUNA_OPENAI_PART_INPUT
        12 => LUNA_OPENAI_PART_CACHED
        14 => LUNA_OPENAI_PART_OUTPUT
        16 => LUNA_OPENAI_PART_TOTAL
        _ => LUNA_OPENAI_PART_END
      }
    LUNA_OPENAI_EVENT_FAILED =>
      match part {
        0 | 3 | 5 | 7 | 9 => LUNA_OPENAI_PART_LITERAL
        1 => LUNA_OPENAI_PART_ID_PREFIX
        2 => LUNA_OPENAI_PART_SEQUENCE
        4 => LUNA_OPENAI_PART_MODEL
        6 => LUNA_OPENAI_PART_PAYLOAD
        8 => LUNA_OPENAI_PART_RETRYABLE
        _ => LUNA_OPENAI_PART_END
      }
    _ => LUNA_OPENAI_PART_END
  }
}

///|
fn luna_openai_part_kind(mode : Int, event_kind : Int, part : Int) -> Int {
  if mode == 1 {
    luna_openai_responses_part(event_kind, part)
  } else {
    luna_openai_chat_part(event_kind, part)
  }
}

///|
fn luna_openai_responses_literal(event_kind : Int, part : Int) -> Bytes {
  match (event_kind, part) {
    (LUNA_OPENAI_EVENT_ACCEPTED, 0) =>
      b"data: {\"type\":\"response.created\",\"response\":{\"id\":\""
    (LUNA_OPENAI_EVENT_ACCEPTED, 3) =>
      b"\",\"object\":\"response\",\"model\":\""
    (LUNA_OPENAI_EVENT_ACCEPTED, 5) => b"\",\"status\":\"in_progress\"}}\n\n"
    (LUNA_OPENAI_EVENT_TOKEN, 0) =>
      b"data: {\"type\":\"response.output_text.delta\",\"response_id\":\""
    (LUNA_OPENAI_EVENT_TOKEN, 3) =>
      b"\",\"output_index\":0,\"content_index\":0,\"delta\":\""
    (LUNA_OPENAI_EVENT_TOKEN, 5) => b"\"}\n\n"
    (LUNA_OPENAI_EVENT_USAGE, 0) =>
      b"data: {\"type\":\"response.usage\",\"response_id\":\""
    (LUNA_OPENAI_EVENT_USAGE, 3) => b"\",\"usage\":{\"input_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 5) => b",\"cached_input_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 7) => b",\"output_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 9) => b",\"total_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 11) => b"}}\n\n"
    (LUNA_OPENAI_EVENT_COMPLETED, 0) =>
      b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 3) =>
      b"\",\"object\":\"response\",\"model\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 5) =>
      b"\",\"status\":\"completed\",\"finish_reason\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 7) => b"\",\"output_text\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 9) => b"\",\"usage\":{\"input_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 11) => b",\"cached_input_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 13) => b",\"output_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 15) => b",\"total_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 17) => b"}}}\n\ndata: [DONE]\n\n"
    (LUNA_OPENAI_EVENT_FAILED, 0) =>
      b"data: {\"type\":\"response.failed\",\"response\":{\"id\":\""
    (LUNA_OPENAI_EVENT_FAILED, 3) =>
      b"\",\"status\":\"failed\",\"error\":{\"code\":\""
    (LUNA_OPENAI_EVENT_FAILED, 5) =>
      b"\",\"message\":\"request failed\",\"retryable\":"
    (LUNA_OPENAI_EVENT_FAILED, 7) => b"}}}\n\ndata: [DONE]\n\n"
    _ => b""
  }
}

///|
fn luna_openai_chat_literal(event_kind : Int, part : Int) -> Bytes {
  match (event_kind, part) {
    (LUNA_OPENAI_EVENT_ACCEPTED, 0) => b"data: {\"id\":\""
    (LUNA_OPENAI_EVENT_ACCEPTED, 3) =>
      b"\",\"object\":\"chat.completion.chunk\",\"model\":\""
    (LUNA_OPENAI_EVENT_ACCEPTED, 5) =>
      b"\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
    (LUNA_OPENAI_EVENT_TOKEN, 0) => b"data: {\"id\":\""
    (LUNA_OPENAI_EVENT_TOKEN, 3) =>
      b"\",\"object\":\"chat.completion.chunk\",\"model\":\""
    (LUNA_OPENAI_EVENT_TOKEN, 5) =>
      b"\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
    (LUNA_OPENAI_EVENT_TOKEN, 7) => b"\"},\"finish_reason\":null}]}\n\n"
    (LUNA_OPENAI_EVENT_USAGE, 0) => b"data: {\"id\":\""
    (LUNA_OPENAI_EVENT_USAGE, 3) =>
      b"\",\"object\":\"chat.completion.chunk\",\"model\":\""
    (LUNA_OPENAI_EVENT_USAGE, 5) =>
      b"\",\"choices\":[],\"usage\":{\"prompt_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 7) => b",\"cached_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 9) => b",\"completion_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 11) => b",\"total_tokens\":"
    (LUNA_OPENAI_EVENT_USAGE, 13) => b"}}\n\n"
    (LUNA_OPENAI_EVENT_COMPLETED, 0) => b"data: {\"id\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 3) =>
      b"\",\"object\":\"chat.completion.chunk\",\"model\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 5) =>
      b"\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 7) => b"\"},\"finish_reason\":\""
    (LUNA_OPENAI_EVENT_COMPLETED, 9) => b"\"}],\"usage\":{\"prompt_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 11) => b",\"cached_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 13) => b",\"completion_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 15) => b",\"total_tokens\":"
    (LUNA_OPENAI_EVENT_COMPLETED, 17) => b"}}\n\ndata: [DONE]\n\n"
    (LUNA_OPENAI_EVENT_FAILED, 0) => b"data: {\"id\":\""
    (LUNA_OPENAI_EVENT_FAILED, 3) =>
      b"\",\"object\":\"chat.completion.chunk\",\"model\":\""
    (LUNA_OPENAI_EVENT_FAILED, 5) => b"\",\"error\":{\"code\":\""
    (LUNA_OPENAI_EVENT_FAILED, 7) =>
      b"\",\"message\":\"request failed\",\"retryable\":"
    (LUNA_OPENAI_EVENT_FAILED, 9) =>
      b",\"type\":\"lunaflux_error\"}}\n\ndata: [DONE]\n\n"
    _ => b""
  }
}

///|
fn luna_openai_literal(mode : Int, event_kind : Int, part : Int) -> Bytes {
  if mode == 1 {
    luna_openai_responses_literal(event_kind, part)
  } else {
    luna_openai_chat_literal(event_kind, part)
  }
}
