///|
extern "C" fn hold_completion(held : Bool) -> Unit = "lunaflux_device_worker_fake_hold_completion"

///|
async fn media_chat_post(
  url : String,
  body : Bytes,
  authorized~ : Bool = true,
) -> (Int32, Json) {
  let client = @http.Client(url)
  defer client.close()
  let response = client.post("/v1/chat/completions", body, extra_headers={
    "Authorization": if authorized {
      "Bearer secret"
    } else {
      "Bearer invalid"
    },
    "Content-Type": "application/json",
    "Content-Length": body.length().to_string(),
  })
  (response.code, client.read_all().json())
}

///|
async fn media_chat_raw_rejection(
  address : @socket.Addr,
  body : Bytes,
  expected : Int32,
  authorized~ : Bool = true,
) -> Unit {
  let peer = @socket.Tcp::connect(address)
  defer peer.close()
  let authorization = if authorized { "secret" } else { "invalid" }
  // Send the bounded fixture in one write: early 401/429 may close immediately.
  peer.write(
    @utf8.encode(
      "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer \{authorization}\r\nContent-Type: application/json\r\nContent-Length: \{body.length()}\r\n\r\n",
    ) +
    body,
  )
  guard peer.read_until("\r\n") is Some(line) else {
    fail("missing HTTP rejection")
  }
  assert_true(line.contains(" \{expected} "))
}

///|
async fn run_media_chat_fixture(
  spec : @model.Spec,
  vision : @vision.Program,
  execution : @image_runtime.Execution,
) -> Unit {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(assets) else {
    println("SKIP actual media HTTP ABI: set LUNA_MINICPM_V47_ASSET_DIR")
    return
  }
  let tokenizer = @prompt.prepare_tokenizer(
    spec,
    @fs.read_file(assets + "/tokenizer.json").binary()[:],
    @tokenizer.TokenizerDigest::parse("0".repeat(64)),
    {
      max_input_bytes: 32768,
      max_output_tokens: 4096,
      max_decoded_bytes: 16777216,
      max_vocab_entries: 250000,
      max_merge_rules: 500000,
      max_token_bytes: 2048,
      max_special_tokens: 128,
    },
  )
  let backend = @chat.Backend::with_images(spec, tokenizer, vision, execution)
  let key = @utf8.encode("secret").to_fixedarray()
  let policy = @auth.LunaApiAuthPolicy::new(
    key,
    source_offset=0,
    length=key.length(),
    maximum_credential_bytes=64,
  )
  defer policy.close()
  let service = backend.service(policy, timeout_ms=3000)
  defer service.close()
  let png = @base64.encode(
    @fs.read_file("media/rgb8/fixtures/rgb.png").binary()[:],
  )
  let jpeg = @base64.encode(
    @fs.read_file("media/rgb8/fixtures/baseline.jpg").binary()[:],
  )
  let image_request : Json = {
    "model": "MiniCPM-V-4.7-35B-A3B",
    "stream": false,
    "temperature": 0,
    "max_tokens": 1,
    "messages": [
      { "role": "system", "content": " policy " },
      {
        "role": "user",
        "content": [
          { "type": "text", "text": " compare " },
          {
            "type": "image_url",
            "image_url": { "url": Json("data:image/png;base64," + png) },
          },
          {
            "type": "image_url",
            "image_url": {
              "url": Json("data:image/jpeg;base64," + jpeg),
              "detail": "high",
            },
          },
        ],
      },
    ],
  }
  let bytes = @utf8.encode(image_request.stringify())
  let normalized = @media.normalize_chat(spec, bytes)
  let prepared = @media.prepare(
    spec,
    tokenizer,
    vision,
    normalized.system(),
    normalized.content()[:],
    maximum_tokens=4095,
    maximum_prepared_pixel_bytes=10000000L,
    system_present=normalized.system_present(),
  )
  let expected_tokens = prepared.prompt().tokens().length()
  let token = tokenizer.encode_text(
      "ok",
      special_policy=AsOrdinary,
      output_limit=16,
      truncation=RejectOverflow,
    ).token_ids[0]
  let piece = tokenizer.decode_text([token], special_policy=Preserve)
  @async.with_timeout(15000, () => {
    @async.with_task_group() <| group => {
      let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
      defer listener.close()
      group.spawn_bg(no_wait=true, () => service.serve(listener))
      let url = "http://127.0.0.1:\{listener.addr.port()}"
      let before = launches()
      set_sample_token(token)
      let (code, response) = media_chat_post(url, bytes)
      assert_eq(code, 200)
      assert_true(
        response
        is {
          "choices": [
            {
              "finish_reason": "length",
              "message": { "content": String(text), .. },
              ..
            },
          ],
          "usage": {
            "prompt_tokens": Number(count, ..),
            "completion_tokens": 1,
            ..
          },
          ..
        } &&
        text == piece &&
        count.to_int() == expected_tokens,
      )
      assert_eq(
        launches() - before,
        318UL + ((expected_tokens + 31) / 32).to_uint64() * 925UL,
      )
      while !service.is_ready() {
        @async.pause()
      }
      let streaming_bytes = @utf8.encode(
        image_request
        .stringify()
        .replace(old="\"stream\":false", new="\"stream\":true"),
      )
      let stream_client = @http.Client(url)
      defer stream_client.close()
      let stream_response = stream_client.post(
        "/v1/chat/completions",
        streaming_bytes,
        extra_headers={
          "Authorization": "Bearer secret",
          "Content-Type": "application/json",
          "Content-Length": streaming_bytes.length().to_string(),
        },
      )
      assert_eq(stream_response.code, 200)
      let (stream_text, terminal) = read_chat_stream(stream_client)
      assert_eq(stream_text, piece)
      assert_true(
        terminal
        is {
          "choices": [{ "finish_reason": "length", .. }],
          "usage": {
            "prompt_tokens": Number(count, ..),
            "completion_tokens": 1,
            ..
          },
          ..
        } &&
        count.to_int() == expected_tokens,
      )
      while !service.is_ready() {
        @async.pause()
      }
      // Same endpoint and execution owner accept no-system text after images.
      let text : Json = {
        "model": "MiniCPM-V-4.7-35B-A3B",
        "temperature": 0,
        "max_tokens": 1,
        "messages": [{ "role": "user", "content": "  hello  " }],
      }
      set_sample_token(248046)
      let (code, response) = media_chat_post(
        url,
        @utf8.encode(text.stringify()),
      )
      assert_eq(code, 200)
      let expected = @prompt.prepare_text_prompt(
          spec,
          tokenizer,
          "",
          "hello",
          maximum_tokens=4095,
          system_present=false,
        )
        .tokens()
        .length()
      assert_true(
        response
        is {
          "choices": [{ "finish_reason": "stop", .. }],
          "usage": { "prompt_tokens": Number(count, ..), .. },
          ..
        } &&
        count.to_int() == expected,
      )
      while !service.is_ready() {
        @async.pause()
      }
      let before_reject = launches()
      println(
        "MiniCPM media HTTP: image output and no-system text usage verified",
      )
      media_chat_raw_rejection(listener.addr, bytes, 401, authorized=false)
      while !service.is_ready() {
        @async.pause()
      }
      let invalid = @utf8.encode(
        image_request
        .stringify()
        .replace(old="data:image/png;base64,", new="https://example.invalid/"),
      )
      assert_eq(media_chat_post(url, invalid).0, 400)
      while !service.is_ready() {
        @async.pause()
      }
      let invalid = @utf8.encode(
        image_request
        .stringify()
        .replace(old="\"temperature\":0", new="\"temperature\":0.5"),
      )
      assert_eq(media_chat_post(url, invalid).0, 400)
      while !service.is_ready() {
        @async.pause()
      }
      assert_eq(launches(), before_reject)
      set_fault(1)
      assert_eq(media_chat_post(url, bytes).0, 503)
      while !service.is_ready() {
        @async.pause()
      }
      assert_true(backend.is_ready())
      set_sample_token(248046)
      assert_eq(media_chat_post(url, bytes).0, 200)
      while !service.is_ready() {
        @async.pause()
      }
      assert_true(execution.is_finished())
      println(
        "MiniCPM media HTTP: authorization/validation and failure recovery verified",
      )
      // Hold the vision completion event: text must not start while pending.
      hold_completion(true)
      defer hold_completion(false)
      let before_pending = launches()
      let pending = group.spawn(() => media_chat_post(url, bytes).0)
      while launches() == before_pending {
        @async.pause()
      }
      assert_eq(launches() - before_pending, 318UL)
      media_chat_raw_rejection(listener.addr, bytes, 429)
      assert_eq(pending.wait(), 503)
      while !service.is_ready() {
        @async.pause()
      }
      assert_true(backend.is_ready())
      assert_true(execution.is_finished())
      assert_eq(execution.finish_reason(), Some(Cancelled))
      println(
        "MiniCPM media HTTP: pending vision timeout/capacity cleanup verified",
      )
      hold_completion(false)
      assert_eq(media_chat_post(url, bytes).0, 200)
      while !service.is_ready() {
        @async.pause()
      }
      // Streaming publishes Accepted while vision is pending. Disconnect after
      // those headers must still retire native work before releasing credit.
      hold_completion(true)
      let stream_peer = @http.Client(url)
      let before_stream_disconnect = launches()
      let accepted = stream_peer.post("/v1/chat/completions", streaming_bytes, extra_headers={
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "Content-Length": streaming_bytes.length().to_string(),
      })
      assert_eq(accepted.code, 200)
      guard stream_peer.read_until("\n\n") is Some(frame) else {
        fail("streaming image Accepted missing")
      }
      assert_true(frame.contains("\"role\":\"assistant\""))
      assert_eq(launches() - before_stream_disconnect, 318UL)
      stream_peer.close()
      while !service.is_ready() {
        @async.pause()
      }
      assert_true(backend.is_ready() && execution.is_finished())
      assert_eq(execution.finish_reason(), Some(Cancelled))
      hold_completion(false)
      assert_eq(media_chat_post(url, bytes).0, 200)
      while !service.is_ready() {
        @async.pause()
      }
      hold_completion(true)
      let peer = @socket.Tcp::connect(listener.addr)
      let before_disconnect = launches()
      peer.write(
        @utf8.encode(
          "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: \{bytes.length()}\r\n\r\n",
        ),
      )
      peer.write(bytes)
      while launches() == before_disconnect {
        @async.pause()
      }
      assert_eq(launches() - before_disconnect, 318UL)
      peer.close()
      while !service.is_ready() {
        @async.pause()
      }
      assert_true(backend.is_ready())
      assert_true(execution.is_finished())
      hold_completion(false)
      assert_eq(media_chat_post(url, bytes).0, 200)
      while !service.is_ready() {
        @async.pause()
      }
    }
  })
  println(
    "MiniCPM-V 4.7 media HTTP ABI: buffered/streaming inline PNG/JPEG, actual tokenizer, canonical admission/usage/stops, no-system text reuse, 401/400/429, pending-vision timeout/disconnect including after SSE headers, and enqueue failure/recovery passed; scripted driver only",
  )
}
