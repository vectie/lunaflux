///|
fn bridge_runtime_error_message(error : BridgeRuntimeError) -> String {
  match error {
    InvalidArgument(message)
    | StartupIdentity(message)
    | HttpRequest(message)
    | NativeProtocol(message) => message
  }
}

///|
async fn bridge_generate(
  http : @socket.Tcp,
  config : BridgeConfig,
  request : @bridge.Qwen3TokenIdRequest,
  request_id : UInt64,
  response_started : Ref[Bool],
) -> Unit raise BridgeRuntimeError {
  let limits = bridge_limits(config)
  let frame = bridge_request_frame(config, request, request_id, limits)
  let native = @socket.Tcp::connect(config.runtime_addr) catch {
    _ => raise NativeProtocol("persistent native service is unavailable")
  }
  defer native.close()
  bridge_write_all(native, frame)
  let framed = @framed_wire.FramedWireLimits::new(
    limits,
    max_frame_bytes=1048576,
  ) catch {
    _ => raise NativeProtocol("native event limits are invalid")
  }
  let decoder = @framed_wire.EventFrameBuffer::new(framed)
  let storage = FixedArray::make(decoder.capacity(), b'\x00')
  let detokenizer = @bridge.Qwen3IncrementalDetokenizer::new(config.tokenizer) catch {
    _ => raise NativeProtocol("Qwen tokenizer decode state is invalid")
  }
  let sse = @bridge.Qwen3TokenSseBuffer::new(
    max_text_bytes=detokenizer.max_delta_bytes(),
  )
  guard bridge_read_event(native, decoder, storage) is Accepted(_) else {
    raise NativeProtocol("native request was not accepted")
  }
  bridge_write_all(
    http, b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n",
  )
  response_started.val = true
  let mut token_count = 0
  let mut usage_count : Int? = None
  for ;; {
    match bridge_read_event(native, decoder, storage) {
      Token(token) => {
        if token.position() != token_count ||
          token_count >= request.max_output_tokens() {
          raise NativeProtocol("native token sequence is invalid")
        }
        let text = detokenizer.advance(token.token_id()) catch {
          _ => raise NativeProtocol("native token cannot be decoded")
        }
        bridge_write_all(http, sse.stage(token.token_id(), text))
        token_count += 1
      }
      Usage(usage) => {
        if usage.input_tokens() != request.input_token_ids().length() ||
          usage.output_tokens() != token_count {
          raise NativeProtocol("native usage differs")
        }
        usage_count = Some(usage.output_tokens())
      }
      Completed(completed) => {
        if usage_count != Some(token_count) ||
          completed.final_usage().output_tokens() != token_count {
          raise NativeProtocol("native terminal usage differs")
        }
        let trailing_text = detokenizer.finish() catch {
          _ => raise NativeProtocol("native terminal text cannot be decoded")
        }
        bridge_write_all(http, sse.terminal(trailing_text))
        return
      }
      Failed(_) | Accepted(_) =>
        raise NativeProtocol("native stream terminated unexpectedly")
    }
  }
}

///|
async fn bridge_handle(
  connection : @socket.Tcp,
  config : BridgeConfig,
  sequence : Ref[UInt64],
) -> Unit {
  let request = bridge_read_http(connection) catch {
    _ => {
      bridge_json_response(
        connection, "400 Bad Request", "{\"status\":\"bad_request\"}\n",
      ) catch {
        _ => ()
      }
      return
    }
  }
  match (request.verb, request.path) {
    ("GET", "/health") =>
      if request.body.is_empty() {
        bridge_json_response(
          connection, "200 OK", "{\"status\":\"healthy\",\"model\":\"Qwen3-0.6B\"}\n",
        ) catch {
          _ => ()
        }
      } else {
        bridge_json_response(
          connection, "400 Bad Request", "{\"status\":\"bad_request\"}\n",
        ) catch {
          _ => ()
        }
      }
    ("POST", "/benchmark/v1/token-ids") => {
      let benchmark = @bridge.parse_qwen3_token_id_request(
        request.body,
        config.max_input_tokens,
        config.max_output_tokens,
        config.max_token_id,
      ) catch {
        _ => {
          bridge_json_response(
            connection, "400 Bad Request", "{\"status\":\"bad_request\"}\n",
          ) catch {
            _ => ()
          }
          return
        }
      }
      let id = sequence.val
      sequence.val += 1UL
      let response_started = Ref(false)
      bridge_generate(connection, config, benchmark, id, response_started) catch {
        error => {
          let message = bridge_runtime_error_message(error)
          println("bridge request failed: " + message)
          // Before the SSE head this reports a bounded upstream failure. If
          // streaming already began, the closed connection is the only honest
          // failure signal; a fake terminal event would corrupt timings.
          if !response_started.val {
            bridge_json_response(
              connection, "502 Bad Gateway", "{\"status\":\"upstream_failed\"}\n",
            ) catch {
              _ => ()
            }
          } else {
            bridge_write_all(
              connection,
              @utf8.encode("event: error\ndata: " + message + "\n\n"),
            ) catch {
              _ => ()
            }
          }
        }
      }
    }
    _ =>
      bridge_json_response(
        connection, "404 Not Found", "{\"status\":\"not_found\"}\n",
      ) catch {
        _ => ()
      }
  }
}

///|
async fn bridge_run(config : BridgeConfig) -> Unit {
  let server = @socket.TcpServer(
    config.listen_addr,
    dual_stack=false,
    reuse_addr=true,
  )
  let sequence = Ref(1UL)
  println("schema=lunaflux.qwen3-token-id-sse-bridge.v1")
  println("model=Qwen3-0.6B")
  println("listen_origin=http://" + server.addr.to_string())
  println("health_path=/health")
  println("benchmark_path=/benchmark/v1/token-ids")
  println("upstream_protocol=native-framed-v1")
  println("max_connections=32")
  println("readiness=true")
  server.run_forever(
    (connection, _) => bridge_handle(connection, config, sequence),
    allow_failure=false,
    max_connections=32,
  )
}
