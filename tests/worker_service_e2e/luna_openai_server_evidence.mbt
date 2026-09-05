///|
fn openai_e2e_bytes(text : String) -> FixedArray[Byte] {
  let encoded = @utf8.encode(text)
  FixedArray::makei(encoded.length(), index => encoded[index])
}

///|
fn openai_e2e_template(
  inference : @inference.InferenceLimits,
) -> @openai_compat.LunaOpenAIChatTemplate raise {
  let empty : FixedArray[Byte] = []
  @openai_compat.LunaOpenAIChatTemplate::new(
    inference,
    empty,
    empty,
    empty,
    empty,
    empty,
    empty,
    empty,
    max_rendered_prompt_bytes=4,
  )
}

///|
fn prepare_openai_e2e_server(
  executable : Bytes,
  deadline_millis? : UInt64 = 10000UL,
  fixture_root? : String = "/private/tmp",
) -> @online_tcp.LunaOnlineOpenAIServerPreparation raise {
  let inference = tcp_inference_limits()
  let model = openai_e2e_bytes("model-x")
  let prefix = openai_e2e_bytes("resp_")
  let secret = openai_e2e_bytes("secret")
  let policy = @api_auth.LunaApiAuthPolicy::new(
    secret,
    source_offset=0,
    length=secret.length(),
    maximum_credential_bytes=8,
  )
  @online_tcp.prepare_luna_online_openai_server(
    prepare_tcp_service(executable, session_tokenizer(), fixture_root~),
    @http1.LunaHttp1Limits::new(max_head_bytes=1024, max_body_bytes=512),
    @http1.LunaHttp1StepBudget::new(1),
    @http1.LunaHttp1ResponseStepBudget::new(17),
    policy,
    @openai_compat.LunaOpenAICompatStepBudget::new(1),
    @openai_compat.LunaOpenAICompatStepBudget::new(17),
    @openai_compat.LunaOpenAIInboundLimits::new(max_messages=2),
    openai_e2e_template(inference),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=prefix.length(),
    @inference.CacheScope::from_ascii(b"scope", inference),
    max_new_tokens=1,
    context_ceiling=4,
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(7UL),
    @inference.DeadlineBudget::from_milliseconds(deadline_millis, inference),
  )
}

///|
fn openai_http_request(
  path : String,
  credential : String,
  body : String,
) -> Bytes {
  let body_bytes = @utf8.encode(body)
  @utf8.encode(
    "POST " +
    path +
    " HTTP/1.1\r\nAuthorization: Bearer " +
    credential +
    "\r\nContent-Length: " +
    body_bytes.length().to_string() +
    "\r\nContent-Type: application/json\r\n\r\n" +
    body,
  )
}

///|
async fn openai_write_range(
  connection : @socket.Tcp,
  source : Bytes,
  offset : Int,
  length : Int,
) -> Unit {
  let mut cursor = 0
  while cursor < length {
    let written = connection.write_once(
      source,
      offset=offset + cursor,
      len=length - cursor,
    )
    if written <= 0 || written > length - cursor {
      raise Failure("OpenAI loopback client write failed")
    }
    cursor += written
  }
}

///|
async fn openai_read_all(connection : @socket.Tcp) -> String {
  let storage = FixedArray::make(16384, b'\x00')
  let mut cursor = 0
  let mut complete = false
  while !complete {
    if cursor == storage.length() {
      raise Failure("OpenAI loopback response exceeded evidence capacity")
    }
    let count = connection.read(
      storage,
      offset=cursor,
      max_len=Int::min(17, storage.length() - cursor),
    )
    if count < 0 || count > storage.length() - cursor {
      raise Failure("OpenAI loopback response returned an invalid count")
    }
    if count == 0 {
      complete = true
    } else {
      cursor += count
    }
  }
  @utf8.decode(Bytes::makei(cursor, index => storage[index]))
}

///|
async fn progress_openai_to_first_observation(
  server : @online_tcp.LunaOnlineOpenAIServer,
  ordinal : UInt64,
) -> Unit {
  let mut attempts = 0
  while attempts < 1000000 {
    match server.progress_on_reactor() {
      LunaOnlineTcpServerObservationRecorded => {
        let metrics = server.metrics_snapshot()
        assert_eq(metrics.counter_value(Admissions), ordinal)
        assert_eq(metrics.counter_value(PromptTokens), (ordinal - 1UL) * 2UL)
        let log = server.log_snapshot()
        assert_true(log.event_at(log.length() - 1) == LunaAdmissionAccepted)
        return
      }
      LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      _ => raise Failure("OpenAI server retired before Admission observation")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("OpenAI server did not publish Admission observation")
}

///|
async fn progress_openai_to_retired(
  server : @online_tcp.LunaOnlineOpenAIServer,
  expected_observations : Int,
  ordinal? : UInt64 = 0UL,
) -> Int {
  let mut observations = 0
  let mut attempts = 0
  while attempts < 1000000 {
    match server.progress_on_reactor() {
      LunaOnlineTcpServerObservationRecorded => {
        observations += 1
        if ordinal > 0UL {
          let metrics = server.metrics_snapshot()
          if observations == 1 {
            // The first-token latency pulse is independently ACKed and does
            // not mutate usage/completion counters.
            assert_eq(
              metrics.counter_value(PromptTokens),
              (ordinal - 1UL) * 2UL,
            )
            assert_eq(metrics.counter_value(GeneratedTokens), ordinal - 1UL)
            assert_eq(metrics.counter_value(Completions), ordinal - 1UL)
          } else if observations == 2 {
            assert_eq(metrics.counter_value(PromptTokens), ordinal * 2UL)
            assert_eq(metrics.counter_value(GeneratedTokens), ordinal)
            assert_eq(metrics.counter_value(Completions), ordinal - 1UL)
          } else if observations == 3 {
            assert_eq(metrics.counter_value(Completions), ordinal)
            let log = server.log_snapshot()
            assert_true(log.event_at(log.length() - 1) == LunaRequestCompleted)
          }
        }
      }
      LunaOnlineTcpServerConnectionRetired => {
        assert_eq(observations, expected_observations)
        return observations
      }
      LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerBackpressured
      | LunaOnlineTcpServerConnectedProgress => ()
      _ => raise Failure("OpenAI server left connected retirement path")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.pause()
    }
  }
  raise Failure("OpenAI connection did not retire")
}

///|
async fn drain_openai_server(
  server : @online_tcp.LunaOnlineOpenAIServer,
) -> Unit {
  server.begin_drain()
  let mut attempts = 0
  while server.state() != LunaOnlineTcpServerClosed && attempts < 1000000 {
    let progress = server.progress_on_reactor()
    match progress {
      LunaOnlineTcpServerDrainingProgress
      | LunaOnlineTcpServerAdvanced
      | LunaOnlineTcpServerClosedProgress => ()
      _ => raise Failure("OpenAI drain reopened connection work")
    }
    attempts += 1
    if attempts % 64 == 0 {
      @async.sleep(1)
    }
  }
  assert_eq(server.state(), LunaOnlineTcpServerClosed)
}

///|
fn assert_openai_error_response(
  response : String,
  status : String,
) -> Unit raise {
  assert_true(response.contains("HTTP/1.1 " + status))
  assert_true(response.contains("Content-Type: application/json"))
  assert_false(response.contains("text/event-stream"))
  assert_false(response.contains("data:"))
}

///|
async fn run_openai_request(
  server : @online_tcp.LunaOnlineOpenAIServer,
  request : Bytes,
  ordinal : UInt64,
) -> String {
  let client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  openai_write_range(client, request, 0, 1)
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerAdvanced)
  openai_write_range(client, request, 1, request.length() - 1)
  progress_openai_to_first_observation(server, ordinal)
  ignore(progress_openai_to_retired(server, 3, ordinal~))
  let response = openai_read_all(client)
  client.close()
  response
}

///|
fn openai_log_reason_count(
  log : @instance_log.LunaInstanceLogSnapshot,
  event : @instance_log.LunaInstanceLogEvent,
  reason : @instance_log.LunaInstanceLogReason,
) -> Int raise {
  let mut count = 0
  for index in 0..<log.length() {
    if log.event_at(index) == event && log.reason_at(index) == reason {
      count += 1
    }
  }
  count
}

///|
fn openai_receipt_deadline_server_limits(
  read_chunk_bytes : Int,
) -> @online_tcp.LunaOnlineTcpLimits raise {
  @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes~,
    write_chunk_bytes=17,
    accept_timeout_millis=25,
    input_idle_timeout_millis=25,
    write_timeout_millis=25,
    reactor_transition_budget=1,
  )
}

///|
async fn assert_openai_receipt_deadline(executable : Bytes) -> Unit {
  let server = @online_tcp.bind_luna_online_openai_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_openai_e2e_server(executable, deadline_millis=50UL),
    openai_receipt_deadline_server_limits(2048),
    log_capacity=16,
  )
  let request = openai_http_request(
    "/v1/responses", "secret", "{\"model\":\"model-x\",\"input\":\"\\u0001\\u0002\",\"stream\":true}",
  )
  let client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  openai_write_range(client, request, 0, 1)
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerAdvanced)
  @async.sleep(75)
  ignore(progress_openai_to_retired(server, 0))
  assert_eq(openai_read_all(client), "")
  client.close()
  drain_openai_server(server)
  assert_eq(server.metrics_snapshot().counter_value(Admissions), 0UL)
  assert_eq(
    openai_log_reason_count(
      server.log_snapshot(),
      LunaNetworkDisconnected,
      LunaLogDeadline,
    ),
    1,
  )
}

///|
async fn run_luna_openai_server_evidence(executable : Bytes) -> Unit {
  let server = @online_tcp.bind_luna_online_openai_server(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_openai_e2e_server(executable),
    reusable_server_limits(2048),
    log_capacity=64,
  )
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerAcceptTimedOut)

  let wrong = openai_http_request(
    "/v1/responses", "wrong", "{\"model\":\"model-x\",\"input\":\"ab\",\"stream\":true}",
  )
  let wrong_client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  openai_write_range(wrong_client, wrong, 0, wrong.length())
  ignore(progress_openai_to_retired(server, 0))
  assert_openai_error_response(openai_read_all(wrong_client), "401")
  wrong_client.close()
  assert_eq(server.metrics_snapshot().counter_value(Admissions), 0UL)
  let wrong_log = server.log_snapshot()
  assert_eq(log_event_count(wrong_log, LunaNetworkAccepted), 1)
  assert_eq(log_event_count(wrong_log, LunaNetworkRejected), 1)
  assert_eq(log_event_count(wrong_log, LunaNetworkDisconnected), 1)
  assert_eq(log_event_count(wrong_log, LunaAdmissionAccepted), 0)
  assert_eq(log_event_count(wrong_log, LunaAdmissionRejected), 0)
  assert_eq(
    openai_log_reason_count(
      wrong_log,
      LunaNetworkRejected,
      LunaLogAuthentication,
    ),
    1,
  )

  let responses = run_openai_request(
    server,
    openai_http_request(
      "/v1/responses", "secret", "{\"model\":\"model-x\",\"input\":\"\\u0001\\u0002\",\"stream\":true}",
    ),
    1UL,
  )
  assert_true(responses.contains("HTTP/1.1 200"))
  assert_true(responses.contains("Content-Type: text/event-stream"))
  assert_true(responses.contains("response.created"))
  assert_true(responses.contains("response.output_text.delta"))
  assert_true(responses.contains("response.usage"))
  assert_true(responses.contains("response.completed"))
  assert_true(responses.contains("data: [DONE]"))

  let chat = run_openai_request(
    server,
    openai_http_request(
      "/v1/chat/completions", "secret", "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"\\u0003\\u0004\"}]}",
    ),
    2UL,
  )
  assert_true(chat.contains("HTTP/1.1 200"))
  assert_true(chat.contains("chat.completion.chunk"))
  assert_true(chat.contains("\"role\":\"assistant\""))
  assert_true(chat.contains("\"usage\""))
  assert_true(chat.contains("data: [DONE]"))

  let invalid = openai_http_request(
    "/v1/responses", "secret", "{\"model\":\"model-x\",\"input\":",
  )
  let invalid_client = @socket.Tcp::connect(server.local_addr())
  assert_eq(server.progress_on_reactor(), LunaOnlineTcpServerConnectedProgress)
  openai_write_range(invalid_client, invalid, 0, invalid.length())
  ignore(progress_openai_to_retired(server, 0))
  assert_openai_error_response(openai_read_all(invalid_client), "400")
  invalid_client.close()

  run_operational_http_evidence(server)
  let metrics = server.metrics_snapshot()
  assert_eq(metrics.counter_value(NetworkAccepts), 4UL)
  assert_eq(metrics.counter_value(NetworkDisconnects), 4UL)
  assert_eq(metrics.counter_value(NetworkRejections), 2UL)
  assert_eq(metrics.counter_value(Admissions), 2UL)
  assert_eq(metrics.counter_value(Completions), 2UL)
  assert_eq(metrics.counter_value(PromptTokens), 4UL)
  assert_eq(metrics.counter_value(GeneratedTokens), 2UL)
  let log = server.log_snapshot()
  assert_eq(log_event_count(log, LunaInstanceReady), 1)
  assert_eq(log_event_count(log, LunaNetworkAccepted), 4)
  assert_eq(log_event_count(log, LunaNetworkRejected), 2)
  assert_eq(log_event_count(log, LunaAdmissionRejected), 0)
  assert_eq(log_event_count(log, LunaAdmissionAccepted), 2)
  assert_eq(log_event_count(log, LunaRequestCompleted), 2)
  assert_eq(log_event_count(log, LunaDrainStarted), 1)
  assert_eq(log_event_count(log, LunaInstanceClosed), 1)
  assert_openai_receipt_deadline(executable)
}
