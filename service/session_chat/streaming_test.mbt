///|
async fn next_sse(client : @http.Client) -> Json {
  guard client.read_until("\n\n") is Some(frame) &&
    frame.trim().strip_prefix("data: ") is Some(payload) else {
    fail("missing SSE JSON frame")
  }
  @json.parse(payload)
}

///|
async test "streaming sends deltas before completion and retires failures timeout and disconnect" {
  @async.with_timeout(8000, () => {
    @async.with_task_group() <| group => {
      let mode = Ref(0)
      let complete = Ref(false)
      let retired = Ref(0)
      let (service, policy) = fixture(400, (handoff, events, emit) => {
        defer {
          retired.val += 1
        }
        events.publish_accepted(
          handoff.model_identity(),
          handoff.effective_limits(),
        )
        publish(events, emit)
        let text = @utf8.encode("你好\n\"robot\"").to_fixedarray()
        assert_true(
          events.publish_token_status(
            token_id=7,
            position=0,
            text,
            source_offset=0,
            length=text.length(),
          )
          is LunaEventPublishReady,
        )
        publish(events, emit)
        if mode.val == 2 || mode.val == 3 {
          @async.sleep(2000)
        }
        if mode.val == 1 {
          let code = @utf8.encode("backend_failure").to_fixedarray()
          events.publish_failed(
            code,
            code_offset=0,
            code_length=code.length(),
            retryable=true,
          )
        } else {
          while !complete.val {
            @async.pause()
          }
          events.publish_completed(
            if mode.val == 4 {
              TokenLimit
            } else {
              StopSequence
            },
            input_tokens=5,
            cached_input_tokens=0,
            output_tokens=1,
            [b'!'],
            tail_offset=0,
            tail_length=1,
          )
        }
        publish(events, emit)
      })
      defer policy.close()
      defer service.close()
      let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
      defer listener.close()
      group.spawn_bg(no_wait=true, () => service.serve(listener))
      let url = "http://127.0.0.1:\{listener.addr.port()}"
      for index in 0..<5 {
        mode.val = index
        complete.val = false
        let client = @http.Client(url)
        defer client.close()
        let body = @utf8.encode(request(streaming=true).stringify())
        let response = client.post("/v1/chat/completions", body, extra_headers={
          "Authorization": "Bearer secret",
          "Content-Type": "application/json",
          "Content-Length": body.length().to_string(),
        })
        assert_eq(response.code, 200)
        assert_true(
          next_sse(client)
          is {
            "object": "chat.completion.chunk",
            "choices": [{ "delta": { "role": "assistant", .. }, .. }],
            ..
          },
        )
        assert_true(
          next_sse(client)
          is {
            "choices": [
              {
                "delta": { "content": "你好\n\"robot\"", .. },
                "finish_reason": Null,
                ..
              },
            ],
            ..
          },
        )
        if index == 0 {
          // Completion is held by this client: a fully buffered implementation
          // cannot reach this point. Keep the same single-request credit.
          assert_eq(retired.val, 0)
          let (busy, _) = post(url + "/v1/chat/completions", request())
          assert_eq(busy.code, 429)
        }
        if index == 3 {
          client.close()
        } else {
          complete.val = true
          let remaining = client.read_all().text()
          if index == 2 {
            // Headers were already sent: timeout closes the stream. It must
            // not append a second HTTP/JSON response or a false success marker.
            assert_eq(remaining, "")
          } else {
            assert_eq(remaining.split("data: [DONE]").to_array().length(), 2)
            assert_false(remaining.contains("HTTP/1.1"))
            if index == 1 {
              assert_true(remaining.contains("\"code\":\"backend_failure\""))
            } else {
              let expected = if index == 4 { "length" } else { "stop" }
              assert_true(
                remaining.contains("\"finish_reason\":\"\{expected}\""),
              )
              assert_true(remaining.contains("\"content\":\"!\""))
              assert_true(remaining.contains("\"completion_tokens\":1"))
            }
          }
        }
        while retired.val != index + 1 || !service.is_ready() {
          @async.pause()
        }
      }
      // Reusing the same renderer after five streams still supports JSON.
      mode.val = 0
      complete.val = true
      let (response, body) = post(url + "/v1/chat/completions", request())
      assert_eq(response.code, 200)
      assert_true(
        body
        is {
          "choices": [
            { "message": { "content": "你好\n\"robot\"!", .. }, .. },
          ],
          ..
        },
      )
    }
  })
}
