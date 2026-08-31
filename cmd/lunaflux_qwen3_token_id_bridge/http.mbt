///|
fn bridge_header_end(bytes : FixedArray[Byte], length : Int) -> Int? {
  if length < 4 {
    return None
  }
  for index in 0..<(length - 3) {
    if bytes[index] == b'\r' &&
      bytes[index + 1] == b'\n' &&
      bytes[index + 2] == b'\r' &&
      bytes[index + 3] == b'\n' {
      return Some(index + 4)
    }
  }
  None
}

///|
fn bridge_http_head(
  bytes : FixedArray[Byte],
  head_length : Int,
) -> (String, String, Int) raise BridgeRuntimeError {
  let head = @utf8.decode(Bytes::makei(head_length, index => bytes[index])) catch {
    _ => raise HttpRequest("HTTP head is not UTF-8")
  }
  let lines = head.split("\r\n").to_array()
  guard lines.length() >= 3 else {
    raise HttpRequest("HTTP head is incomplete")
  }
  let request_line = lines[0].split(" ").to_array()
  guard request_line.length() == 3 && request_line[2] == "HTTP/1.1" else {
    raise HttpRequest("HTTP request line is unsupported")
  }
  let mut content_length : Int? = None
  let mut content_type = false
  let mut transfer_encoding = false
  for index in 1..<lines.length() {
    let line = lines[index].to_owned()
    if line.is_empty() {
      continue
    }
    guard line.find(":") is Some(split) && split > 0 else {
      raise HttpRequest("HTTP header is malformed")
    }
    let name = line[:split].to_owned().to_lower()
    let value = line[split + 1:].trim().to_owned().to_lower()
    match name {
      "content-length" => {
        if content_length is Some(_) {
          raise HttpRequest("duplicate content length")
        }
        let parsed = @string.parse_int(value, base=10) catch {
          _ => raise HttpRequest("content length is malformed")
        }
        if parsed < 0 || parsed > 131072 {
          raise HttpRequest("content length is outside bounds")
        }
        content_length = Some(parsed)
      }
      "content-type" =>
        content_type = value == "application/json" ||
          value == "application/json; charset=utf-8"
      "transfer-encoding" => transfer_encoding = true
      _ => ()
    }
  }
  if transfer_encoding {
    raise HttpRequest("transfer encoding is unsupported")
  }
  let verb = request_line[0].to_owned()
  let path = request_line[1].to_owned()
  let length = match (verb, path) {
    ("GET", "/health") => content_length.unwrap_or(0)
    ("POST", "/benchmark/v1/token-ids") => {
      if !content_type || content_length is None {
        raise HttpRequest("benchmark JSON headers are incomplete")
      }
      content_length.unwrap()
    }
    _ => content_length.unwrap_or(0)
  }
  (verb, path, length)
}

///|
async fn bridge_read_http(
  connection : @socket.Tcp,
) -> HttpRequest raise BridgeRuntimeError {
  let storage = FixedArray::make(196608, b'\x00')
  let mut used = 0
  let mut head_end : Int? = None
  let mut body_length = 0
  let mut verb = ""
  let mut path = ""
  while head_end is None {
    if used >= 65536 {
      raise HttpRequest("HTTP head exceeds its bound")
    }
    let count = @async.with_timeout_opt(30000, () => {
      connection.read(storage, offset=used, max_len=65536 - used)
    }) catch {
      _ => raise HttpRequest("HTTP head read failed")
    }
    guard count is Some(received) && received > 0 else {
      raise HttpRequest("HTTP head read timed out")
    }
    used += received
    head_end = bridge_header_end(storage, used)
    if head_end is Some(end) {
      let parsed = bridge_http_head(storage, end)
      verb = parsed.0
      path = parsed.1
      body_length = parsed.2
    }
  }
  let end = head_end.unwrap()
  if end + body_length > storage.length() {
    raise HttpRequest("HTTP body exceeds its bound")
  }
  while used < end + body_length {
    let count = @async.with_timeout_opt(30000, () => {
      connection.read(storage, offset=used, max_len=end + body_length - used)
    }) catch {
      _ => raise HttpRequest("HTTP body read failed")
    }
    guard count is Some(received) && received > 0 else {
      raise HttpRequest("HTTP body read timed out")
    }
    used += received
  }
  if used != end + body_length {
    raise HttpRequest("HTTP request has trailing bytes")
  }
  let body = @utf8.decode(
    Bytes::makei(body_length, index => storage[end + index]),
  ) catch {
    _ => raise HttpRequest("HTTP body is not UTF-8")
  }
  { verb, path, body }
}

///|
async fn bridge_write_all(
  connection : @socket.Tcp,
  bytes : Bytes,
) -> Unit raise BridgeRuntimeError {
  let mut offset = 0
  while offset < bytes.length() {
    let count = @async.with_timeout_opt(60000, () => {
      connection.write_once(bytes, offset~, len=bytes.length() - offset)
    }) catch {
      _ => raise HttpRequest("response write failed")
    }
    guard count is Some(written) && written > 0 else {
      raise HttpRequest("response write timed out")
    }
    offset += written
  }
}

///|
fn bridge_http_chunk(bytes : Bytes) -> Bytes {
  let prefix = @utf8.encode(bytes.length().to_string(radix=16) + "\r\n")
  let prefix_length = prefix.length()
  let payload_length = bytes.length()
  Bytes::makei(prefix_length + payload_length + 2, index => {
    if index < prefix_length {
      prefix[index]
    } else if index < prefix_length + payload_length {
      bytes[index - prefix_length]
    } else if index == prefix_length + payload_length {
      b'\r'
    } else {
      b'\n'
    }
  })
}

///|
async fn bridge_write_http_chunk(
  connection : @socket.Tcp,
  bytes : Bytes,
) -> Unit raise BridgeRuntimeError {
  bridge_write_all(connection, bridge_http_chunk(bytes))
}

///|
async fn bridge_finish_http_chunks(
  connection : @socket.Tcp,
) -> Unit raise BridgeRuntimeError {
  bridge_write_all(connection, b"0\r\n\r\n")
}

///|
async fn bridge_json_response(
  connection : @socket.Tcp,
  status : String,
  body : String,
) -> Unit raise BridgeRuntimeError {
  let response = "HTTP/1.1 " +
    status +
    "\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: " +
    @utf8.encode(body).length().to_string() +
    "\r\n\r\n" +
    body
  bridge_write_all(connection, @utf8.encode(response))
}
