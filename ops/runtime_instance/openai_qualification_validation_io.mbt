///|
fn qualification_decode(
  source : FixedArray[Byte],
) -> String raise RuntimeOpenAiQualificationValidationError {
  @utf8.decode(Bytes::makei(source.length(), index => source[index])) catch {
    _ =>
      raise openai_qualification_validation_error(OpenAiQualificationContract)
  }
}

///|
fn qualification_wrong_credential(
  source : FixedArray[Byte],
) -> FixedArray[Byte] {
  let wrong = FixedArray::make(source.length(), b'\x00')
  for index in 0..<source.length() {
    wrong[index] = source[index]
  }
  wrong[0] = if wrong[0] == b'A' { b'B' } else { b'A' }
  wrong
}

///|
fn qualification_http_request(credential : String?, body : String) -> Bytes {
  let authorization = match credential {
    Some(value) => "Authorization: Bearer " + value + "\r\n"
    None => ""
  }
  let body_bytes = @utf8.encode(body)
  @utf8.encode(
    "POST /v1/responses HTTP/1.1\r\n" +
    authorization +
    "Content-Length: " +
    body_bytes.length().to_string() +
    "\r\nContent-Type: application/json\r\n\r\n" +
    body,
  )
}

///|
async fn qualification_write_all(
  client : @socket.Tcp,
  source : Bytes,
  timeout_millis : Int,
) -> Unit raise RuntimeOpenAiQualificationValidationError {
  let mut cursor = 0
  while cursor < source.length() {
    let written = @async.with_timeout_opt(timeout_millis, () => {
      client.write_once(source, offset=cursor, len=source.length() - cursor)
    }) catch {
      _ =>
        raise openai_qualification_validation_error(OpenAiQualificationIngress)
    }
    let count = match written {
      Some(value) => value
      None =>
        raise openai_qualification_validation_error(OpenAiQualificationIngress)
    }
    if count <= 0 || count > source.length() - cursor {
      raise openai_qualification_validation_error(OpenAiQualificationIngress)
    }
    cursor += count
  }
}

///|
async fn qualification_read_all(
  client : @socket.Tcp,
  maximum_bytes : Int,
  timeout_millis : Int,
) -> String raise RuntimeOpenAiQualificationValidationError {
  let storage = FixedArray::make(maximum_bytes, b'\x00')
  let mut cursor = 0
  while cursor < storage.length() {
    let received = @async.with_timeout_opt(timeout_millis, () => {
      client.read(storage, offset=cursor, max_len=storage.length() - cursor)
    }) catch {
      _ =>
        raise openai_qualification_validation_error(OpenAiQualificationResponse)
    }
    let count = match received {
      Some(value) => value
      None =>
        raise openai_qualification_validation_error(OpenAiQualificationResponse)
    }
    if count < 0 || count > storage.length() - cursor {
      raise openai_qualification_validation_error(OpenAiQualificationResponse)
    }
    if count == 0 {
      return @utf8.decode(Bytes::makei(cursor, index => storage[index])) catch {
        _ =>
          raise openai_qualification_validation_error(
            OpenAiQualificationResponse,
          )
      }
    }
    cursor += count
  }
  raise openai_qualification_validation_error(OpenAiQualificationResponse)
}

///|
async fn qualification_connect(
  owner : RuntimeInstanceOwner,
  timeout_millis : Int,
) -> @socket.Tcp raise RuntimeOpenAiQualificationValidationError {
  let connected = @async.with_timeout_opt(timeout_millis, () => {
    @socket.Tcp::connect(owner.openai_servers[0].local_addr())
  }) catch {
    _ => raise openai_qualification_validation_error(OpenAiQualificationConnect)
  }
  match connected {
    Some(client) => client
    None =>
      raise openai_qualification_validation_error(OpenAiQualificationConnect)
  }
}

///|
async fn qualification_wait_connected(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
) -> Unit raise RuntimeOpenAiQualificationValidationError {
  for turn in 0..<turn_limit {
    ignore(
      owner.progress() catch {
        _ =>
          raise openai_qualification_validation_error(OpenAiQualificationOwner)
      },
    )
    if owner.phase() != OpenAiQualificationReady ||
      owner.health() != RuntimeHealthy ||
      owner.readiness() != RuntimeNotReady ||
      owner.openai_qualification_readiness() != OpenAiQualificationListenerReady {
      raise openai_qualification_validation_error(OpenAiQualificationOwner)
    }
    if owner.openai_servers[0].state() == LunaOnlineTcpServerConnected {
      return
    }
    if turn % 64 == 63 {
      @async.pause() catch {
        _ =>
          raise openai_qualification_validation_error(OpenAiQualificationOwner)
      }
    }
  }
  raise openai_qualification_validation_error(OpenAiQualificationConnect)
}

///|
async fn qualification_wait_retired(
  owner : RuntimeInstanceOwner,
  expected_disconnects : UInt64,
  turn_limit : Int,
) -> Unit raise RuntimeOpenAiQualificationValidationError {
  for turn in 0..<turn_limit {
    ignore(
      owner.progress() catch {
        _ =>
          raise openai_qualification_validation_error(OpenAiQualificationOwner)
      },
    )
    if owner.phase() != OpenAiQualificationReady ||
      owner.health() != RuntimeHealthy ||
      owner.readiness() != RuntimeNotReady ||
      owner.openai_qualification_readiness() != OpenAiQualificationListenerReady {
      raise openai_qualification_validation_error(OpenAiQualificationOwner)
    }
    let snapshot = owner.metrics_snapshot() catch {
      _ =>
        raise openai_qualification_validation_error(OpenAiQualificationMetrics)
    }
    let disconnected = snapshot.counter_value(NetworkDisconnects) catch {
      _ =>
        raise openai_qualification_validation_error(OpenAiQualificationMetrics)
    }
    if owner.openai_servers[0].state() == LunaOnlineTcpServerListening &&
      disconnected == expected_disconnects {
      return
    }
    if turn % 64 == 63 {
      @async.pause() catch {
        _ =>
          raise openai_qualification_validation_error(OpenAiQualificationOwner)
      }
    }
  }
  raise openai_qualification_validation_error(OpenAiQualificationResponse)
}

///|
async fn qualification_exchange(
  owner : RuntimeInstanceOwner,
  request : Bytes,
  expected_disconnects : UInt64,
  contract : RuntimeOpenAiQualificationValidationContract,
) -> String raise RuntimeOpenAiQualificationValidationError {
  let client = qualification_connect(owner, contract.io_timeout_millis)
  let mut open = true
  defer (if open { client.close() })
  qualification_wait_connected(owner, contract.turn_limit)
  qualification_write_all(client, request, contract.io_timeout_millis)
  qualification_wait_retired(owner, expected_disconnects, contract.turn_limit)
  let response = qualification_read_all(
    client,
    contract.maximum_response_bytes,
    contract.io_timeout_millis,
  )
  client.close()
  open = false
  response
}

///|
async fn qualification_operational_connect(
  owner : RuntimeInstanceOwner,
  timeout_millis : Int,
) -> @socket.Tcp raise RuntimeOpenAiQualificationValidationError {
  let connected = @async.with_timeout_opt(timeout_millis, () => {
    @socket.Tcp::connect(owner.operational_local_addr())
  }) catch {
    _ => raise openai_qualification_validation_error(OpenAiQualificationConnect)
  }
  match connected {
    Some(client) => client
    None =>
      raise openai_qualification_validation_error(OpenAiQualificationConnect)
  }
}

///|
async fn qualification_wait_operational_state(
  owner : RuntimeInstanceOwner,
  expected : @online_tcp.LunaOnlineTcpServerState,
  turn_limit : Int,
) -> Unit raise RuntimeOpenAiQualificationValidationError {
  for turn in 0..<turn_limit {
    ignore(
      owner.progress() catch {
        _ =>
          raise openai_qualification_validation_error(OpenAiQualificationOwner)
      },
    )
    if owner.phase() != OpenAiQualificationReady ||
      owner.health() != RuntimeHealthy ||
      owner.readiness() != RuntimeNotReady ||
      owner.openai_qualification_readiness() != OpenAiQualificationListenerReady {
      raise openai_qualification_validation_error(OpenAiQualificationOwner)
    }
    if owner.openai_servers.length() == 1 &&
      owner.controls.length() == 0 &&
      owner.openai_servers[0].state() == expected {
      return
    }
    if turn % 64 == 63 {
      @async.pause() catch {
        _ =>
          raise openai_qualification_validation_error(OpenAiQualificationOwner)
      }
    }
  }
  raise openai_qualification_validation_error(OpenAiQualificationResponse)
}

///|
async fn qualification_operational_exchange(
  owner : RuntimeInstanceOwner,
  path : String,
  contract : RuntimeOpenAiQualificationValidationContract,
) -> String raise RuntimeOpenAiQualificationValidationError {
  let client = qualification_operational_connect(
    owner,
    contract.io_timeout_millis,
  )
  let mut open = true
  defer (if open { client.close() })
  qualification_wait_operational_state(
    owner,
    LunaOnlineTcpServerConnected,
    contract.turn_limit,
  )
  qualification_write_all(
    client,
    @utf8.encode("GET " + path + " HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    contract.io_timeout_millis,
  )
  qualification_wait_operational_state(
    owner,
    LunaOnlineTcpServerListening,
    contract.turn_limit,
  )
  let response = qualification_read_all(
    client,
    1024,
    contract.io_timeout_millis,
  )
  client.close()
  open = false
  response
}

///|
fn qualification_find_from(source : Bytes, needle : Bytes, start : Int) -> Int {
  if needle.length() == 0 || start < 0 || needle.length() > source.length() {
    return -1
  }
  let stop = source.length() - needle.length() + 1
  for offset in start..<stop {
    let mut matches = true
    for index in 0..<needle.length() {
      if source[offset + index] != needle[index] {
        matches = false
        break
      }
    }
    if matches {
      return offset
    }
  }
  -1
}

///|
fn qualification_markers_ordered(response : String) -> Bool {
  let source = @utf8.encode(response)
  let markers = [
    @utf8.encode("response.created"),
    @utf8.encode("response.output_text.delta"),
    @utf8.encode("response.usage"),
    @utf8.encode("response.completed"),
    @utf8.encode("data: [DONE]"),
  ]
  let mut cursor = 0
  for marker in markers {
    let found = qualification_find_from(source, marker, cursor)
    if found < 0 {
      return false
    }
    cursor = found + marker.length()
  }
  true
}
