///|
fn bridge_http_head_fixture(
  value : String,
) -> (String, String, Int) raise BridgeRuntimeError {
  let bytes = @utf8.encode(value)
  let storage = FixedArray::make(bytes.length(), b'\x00')
  for index, byte in bytes {
    storage[index] = byte
  }
  bridge_http_head(storage, bytes.length())
}

///|
test "benchmark HTTP head accepts exact endpoint with case-insensitive headers" {
  let parsed = try! bridge_http_head_fixture(
    "POST /benchmark/v1/token-ids HTTP/1.1\r\nHost: 127.0.0.1\r\ncontent-TYPE: application/json\r\nCONTENT-length: 17\r\nAccept: text/event-stream\r\n\r\n",
  )
  assert_eq(parsed.0, "POST")
  assert_eq(parsed.1, "/benchmark/v1/token-ids")
  assert_eq(parsed.2, 17)
}

///|
test "chunked benchmark responses frame payload and terminal boundary" {
  assert_eq(@utf8.decode(bridge_http_chunk(b"abc")), "3\r\nabc\r\n")
  assert_eq(
    @utf8.decode(bridge_http_chunk(b"0123456789abcdef")),
    "10\r\n0123456789abcdef\r\n",
  )
}

///|
test "benchmark HTTP head rejects smuggling and unbounded bodies" {
  for
    hostile in [
      "POST /benchmark/v1/token-ids HTTP/1.1\r\nContent-Type: application/json\r\n\r\n",
      "POST /benchmark/v1/token-ids HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n",
      "POST /benchmark/v1/token-ids HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
      "POST /benchmark/v1/token-ids HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 131073\r\n\r\n",
      "POST /benchmark/v1/token-ids HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n",
    ] {
    try bridge_http_head_fixture(hostile) catch {
      HttpRequest(_) => ()
      _ => fail("unexpected hostile HTTP error")
    } noraise {
      _ => fail("hostile HTTP head was admitted")
    }
  }
}
