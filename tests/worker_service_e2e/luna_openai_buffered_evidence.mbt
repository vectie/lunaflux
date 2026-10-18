///|
async fn run_luna_openai_buffered_evidence(
  executable : Bytes,
  fixture_root : String,
) -> Unit {
  let limits = @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes=2048,
    write_chunk_bytes=256,
    accept_timeout_millis=25,
    input_idle_timeout_millis=60000,
    write_timeout_millis=60000,
    reactor_transition_budget=1,
  )
  let pool = @online_tcp.bind_luna_online_openai_connection_pool(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_openai_pool_e2e(
      executable,
      limits,
      fixture_root~,
      output_buffer_bytes=192,
    ),
    limits,
    log_capacity=64,
  )
  // Every JSON body exceeds the single output cell. Two complete requests
  // occupy distinct routes; then the same slots are reused in streaming mode.
  let first = @socket.Tcp::connect(pool.local_addr())
  progress_openai_pool_to_connections(pool, 1, "buffered-first")
  let second = @socket.Tcp::connect(pool.local_addr())
  progress_openai_pool_to_connections(pool, 2, "buffered-second")
  let requests = [
    (first, "/v1/responses", "{\"model\":\"model-x\",\"input\":\"\\u0002\"}"),
    (
      second, "/v1/chat/completions", "{\"model\":\"model-x\",\"messages\":[{\"role\":\"user\",\"content\":\"\\u0002\"}],\"stream\":false}",
    ),
  ]
  for (connection, route, body) in requests {
    let request = openai_http_request(route, "secret", body)
    openai_write_range(connection, request, 0, request.length())
  }
  progress_openai_pool_to_connections(pool, 0, "buffered-both-retired")
  for (connection, _, _) in requests {
    let response = openai_read_all(connection)
    assert_true(response.has_prefix("HTTP/1.1 200 OK\r\n"))
    assert_true(response.contains("Content-Type: application/json\r\n"))
    assert_false(response.contains("data: "))
    assert_false(response.contains("[DONE]"))
    let sections = response.split("\r\n\r\n").collect()
    assert_eq(sections.length(), 2)
    let body = sections[1].to_owned()
    assert_true(
      sections[0].contains("Content-Length: \{@utf8.encode(body).length()}"),
    )
    assert_true(body.has_prefix("{"))
    assert_true(body.has_suffix("}"))
    assert_true(@utf8.encode(body).length() > 192)
    connection.close()
  }
  let streaming = @socket.Tcp::connect(pool.local_addr())
  progress_openai_pool_to_connections(pool, 1, "stream-reuse")
  let request = openai_http_request(
    "/v1/responses", "secret", "{\"model\":\"model-x\",\"input\":\"\\u0002\",\"stream\":true}",
  )
  openai_write_range(streaming, request, 0, request.length())
  progress_openai_pool_to_connections(pool, 0, "stream-retired")
  let response = openai_read_all(streaming)
  assert_true(response.contains("Content-Type: text/event-stream"))
  assert_true(response.contains("data: [DONE]"))
  streaming.close()
  assert_eq(pool.metrics_snapshot().counter_value(Completions), 3UL)
  assert_eq(pool.metrics_snapshot().gauge_value(ActiveRequests), 0)
  drain_openai_pool(pool)
}
