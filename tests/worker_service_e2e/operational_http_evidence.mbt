///|
fn operational_http_request(text : String) -> Bytes {
  @utf8.encode(text)
}

///|
async fn progress_control_to_retired(
  control : @online_tcp.LunaOnlineHttpControlServer,
) -> Unit {
  let mut attempts = 0
  while attempts < 100000 {
    match control.progress_on_reactor() {
      LunaOnlineHttpControlConnectionRetired => return
      LunaOnlineHttpControlAdvanced | LunaOnlineHttpControlConnectedProgress =>
        ()
      _ => raise Failure("operational HTTP connection did not retire cleanly")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("operational HTTP connection exceeded progress bound")
}

///|
async fn operational_http_exchange(
  control : @online_tcp.LunaOnlineHttpControlServer,
  request : Bytes,
) -> String {
  let client = @socket.Tcp::connect(control.local_addr())
  assert_eq(
    control.progress_on_reactor(),
    LunaOnlineHttpControlConnectedProgress,
  )
  openai_write_range(client, request, 0, request.length())
  progress_control_to_retired(control)
  let response = openai_read_all(client)
  client.close()
  response
}

///|
fn assert_operational_response(
  response : String,
  status : String,
  body : String,
) -> Unit raise {
  assert_true(response.contains("HTTP/1.1 " + status))
  assert_true(response.has_suffix(body))
  assert_true(response.contains("Content-Type: application/json"))
  assert_true(response.contains("Cache-Control: no-store"))
  assert_true(response.contains("Connection: close"))
  assert_false(response.contains("secret"))
  assert_false(response.contains("model-x"))
}

///|
fn[T] report_operational_bind_failure(
  message : String,
  error : Error,
) -> T raise {
  println(message)
  raise error
}

///|
async fn run_operational_http_evidence(
  server : @online_tcp.LunaOnlineOpenAIServer,
) -> Unit {
  let prepared = @online_tcp.prepare_luna_online_http_control_for_openai_production(
    server,
    @http1.LunaHttp1ControlLimits::new(max_head_bytes=256),
  )
  let control = @online_tcp.bind_luna_online_http_control_server(
    prepared,
    @online_tcp.LunaOnlineTcpLimits::new(
      read_chunk_bytes=256,
      write_chunk_bytes=17,
      accept_timeout_millis=25,
      input_idle_timeout_millis=100,
      write_timeout_millis=5000,
      reactor_transition_budget=1,
    ),
  ) catch {
    error =>
      report_operational_bind_failure(
        "LunaFlux operational HTTP control bind failed", error,
      )
  }
  assert_eq(control.health(), LunaOnlineTcpServerHealthy)
  assert_eq(control.readiness(), LunaOnlineHttpControlReady)

  // 2,048 empty turns must not accumulate the old 1 ms accept wait. The
  // generous outer bound distinguishes zero-wait polling from timed polling.
  @async.with_timeout(500, () => {
    for _ in 0..<2048 {
      assert_eq(
        control.progress_on_reactor(),
        LunaOnlineHttpControlAcceptTimedOut,
      )
    }
  })

  // An empty poll retains a connected peer until its real idle deadline.
  let slow_client = @socket.Tcp::connect(control.local_addr())
  assert_eq(
    control.progress_on_reactor(),
    LunaOnlineHttpControlConnectedProgress,
  )
  assert_eq(control.progress_on_reactor(), LunaOnlineHttpControlAdvanced)
  assert_eq(control.state(), LunaOnlineHttpControlConnected)
  @async.sleep(110)
  assert_eq(
    control.progress_on_reactor(),
    LunaOnlineHttpControlConnectionRetired,
  )
  slow_client.close()
  assert_eq(control.health(), LunaOnlineTcpServerHealthy)
  assert_eq(control.readiness(), LunaOnlineHttpControlReady)

  // A fragmented request may pause for longer than the former 1 ms poll cap.
  let fragmented = @socket.Tcp::connect(control.local_addr())
  assert_eq(
    control.progress_on_reactor(),
    LunaOnlineHttpControlConnectedProgress,
  )
  let prefix = b"GET /healthz HTTP/1.1\r\n"
  openai_write_range(fragmented, prefix, 0, prefix.length())
  assert_eq(control.progress_on_reactor(), LunaOnlineHttpControlAdvanced)
  assert_eq(control.progress_on_reactor(), LunaOnlineHttpControlAdvanced)
  @async.sleep(5)
  let suffix = b"Host: localhost\r\n\r\n"
  openai_write_range(fragmented, suffix, 0, suffix.length())
  progress_control_to_retired(control)
  assert_operational_response(
    openai_read_all(fragmented),
    "200 OK",
    "{\"status\":\"healthy\"}\n",
  )
  fragmented.close()

  let health = operational_http_exchange(
    control,
    operational_http_request("GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n"),
  )
  assert_operational_response(health, "200 OK", "{\"status\":\"healthy\"}\n")
  let ready = operational_http_exchange(
    control,
    operational_http_request(
      "GET /readyz HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
    ),
  )
  assert_operational_response(ready, "200 OK", "{\"status\":\"ready\"}\n")

  for
    hostile in [
      "POST /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n", "GET /unknown HTTP/1.1\r\nHost: localhost\r\n\r\n",
      "GET /readyz HTTP/1.1\r\nContent-Length: 1\r\n\r\nx",
    ] {
    let response = operational_http_exchange(
      control,
      operational_http_request(hostile),
    )
    assert_operational_response(
      response, "400 Bad Request", "{\"status\":\"bad_request\"}\n",
    )
  }

  server.begin_drain()
  assert_eq(control.health(), LunaOnlineTcpServerHealthy)
  assert_eq(control.readiness(), LunaOnlineHttpControlNotReady)
  let draining_ready = operational_http_exchange(
    control,
    operational_http_request("GET /readyz HTTP/1.1\r\n\r\n"),
  )
  assert_operational_response(
    draining_ready, "503 Service Unavailable", "{\"status\":\"not_ready\"}\n",
  )
  let draining_health = operational_http_exchange(
    control,
    operational_http_request("GET /healthz HTTP/1.1\r\n\r\n"),
  )
  assert_operational_response(
    draining_health, "200 OK", "{\"status\":\"healthy\"}\n",
  )

  drain_openai_server(server)
  assert_eq(control.health(), LunaOnlineTcpServerFailed)
  let closed_health = operational_http_exchange(
    control,
    operational_http_request("GET /healthz HTTP/1.1\r\n\r\n"),
  )
  assert_operational_response(
    closed_health, "503 Service Unavailable", "{\"status\":\"unhealthy\"}\n",
  )
  control.begin_drain()
  let mut attempts = 0
  while control.state() != LunaOnlineHttpControlClosed && attempts < 4 {
    ignore(control.progress_on_reactor())
    attempts += 1
  }
  assert_eq(control.state(), LunaOnlineHttpControlClosed)
}

///|
async fn run_operational_http_only(
  executable : Bytes,
  fixture_root? : String = "/private/tmp",
) -> Unit {
  let server = @online_tcp.bind_luna_online_openai_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_openai_e2e_server(executable, fixture_root~),
    reusable_server_limits(256),
    log_capacity=32,
  ) catch {
    error =>
      report_operational_bind_failure(
        "LunaFlux operational HTTP source bind failed", error,
      )
  }
  run_operational_http_evidence(server)
  assert_eq(server.state(), LunaOnlineTcpServerClosed)
}
