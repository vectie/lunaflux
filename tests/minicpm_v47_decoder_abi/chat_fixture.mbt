///|
/// Real HTTP/parser/tokenizer/output adapter over the scripted CUDA test double.
/// This validates lifetimes and framing, never actual model computation.
async fn run_chat_fixture(
  model : @model.Spec,
  runner : @request.Runner,
) -> Unit {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(assets) else {
    println("SKIP real-tokenizer chat ABI: set LUNA_MINICPM_V47_ASSET_DIR")
    return
  }
  let tokenizer = @prompt.prepare_tokenizer(
    model,
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
  let backend = @chat.Backend::new(model, tokenizer, runner)
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
  @async.with_timeout(10000, () => {
    @async.with_task_group() <| group => {
      let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
      defer listener.close()
      group.spawn_bg(no_wait=true, () => service.serve(listener))
      let url = "http://127.0.0.1:\{listener.addr.port()}"
      let encoded = tokenizer.encode_text(
        "ok",
        special_policy=AsOrdinary,
        output_limit=16,
        truncation=RejectOverflow,
      )
      let token = encoded.token_ids[0]
      let piece = tokenizer.decode_text([token], special_policy=Preserve)
      let before = launches()
      for streaming in [false, true] {
        for request in 0..<3 {
          let messages : Array[Json] = []
          if request != 2 {
            messages.push({
              "role": "system",
              "content": "\u{00a0}policy\u{3000}",
            })
          }
          let content : Json = if request == 1 {
            [{ "type": "text", "text": "\u{000b}task\u{0085}" }]
          } else {
            "\u{000b}task\u{0085}"
          }
          messages.push({ "role": "user", "content": content })
          let expected_input = @prompt.prepare_text_prompt(
              model,
              tokenizer,
              "policy",
              "task",
              maximum_tokens=4096,
              system_present=request != 2,
            )
            .tokens()
            .length()
          let fields : Map[String, Json] = Map([
            ("model", "MiniCPM-V-4.7-35B-A3B"),
            ("stream", Json(streaming)),
            ("temperature", 0),
            ("max_tokens", Json(if request == 1 { 1 } else { 2 })),
            ("messages", Json(messages)),
          ])
          if request == 2 {
            fields["stop"] = Json(piece)
          }
          let bytes = @utf8.encode(Json(fields).stringify())
          let client = @http.Client(url)
          defer client.close()
          set_sample_token(if request == 0 { 248046 } else { token })
          let response = client.post("/v1/chat/completions", bytes, extra_headers={
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
            "Content-Length": bytes.length().to_string(),
          })
          assert_eq(response.code, 200)
          let expected_reason = if request == 1 { "length" } else { "stop" }
          let expected_text = if request == 1 { piece } else { "" }
          let body = if streaming {
            let (content, terminal) = read_chat_stream(client)
            assert_eq(content, expected_text)
            guard terminal
              is {
                "choices": [{ "finish_reason": reason, .. }],
                "usage": usage,
                ..
              } else {
              fail("streaming terminal usage missing")
            }
            let normalized : Json = {
              "choices": [
                {
                  "finish_reason": reason,
                  "message": { "content": Json(content) },
                },
              ],
              "usage": usage,
            }
            normalized
          } else {
            client.read_all().json()
          }
          assert_true(
            body
            is {
              "choices": [
                {
                  "finish_reason": String(reason),
                  "message": { "content": String(text), .. },
                  ..
                },
              ],
              "usage": {
                "completion_tokens": 1,
                "prompt_tokens": Number(input, ..),
                ..
              },
              ..
            } &&
            reason == expected_reason &&
            text == expected_text &&
            input.to_int() == expected_input,
          )
          while !service.is_ready() {
            @async.pause()
          }
          assert_true(backend.is_ready())
        }
      }
      assert_eq(launches() - before, 5544UL)
      for recovery in 0..<3 {
        let body : Json = {
          "model": "MiniCPM-V-4.7-35B-A3B",
          "stream": false,
          "temperature": Json(if recovery == 0 { 0.5 } else { 0.0 }),
          "max_tokens": 2,
          "messages": [
            { "role": "system", "content": "policy" },
            { "role": "user", "content": "task" },
          ],
        }
        let bytes = @utf8.encode(body.stringify())
        let client = @http.Client(url)
        defer client.close()
        if recovery == 1 {
          set_fault(1)
        }
        set_sample_token(248046)
        let response = client.post("/v1/chat/completions", bytes, extra_headers={
          "Authorization": "Bearer secret",
          "Content-Type": "application/json",
          "Content-Length": bytes.length().to_string(),
        })
        assert_eq(
          response.code,
          match recovery {
            0 => 400
            1 => 503
            _ => 200
          },
        )
        ignore(client.read_all())
        while !service.is_ready() {
          @async.pause()
        }
        assert_true(backend.is_ready())
      }
      assert_eq(launches() - before, 6468UL)
    }
  })
  println(
    "MiniCPM-V 4.7 chat ABI: actual tokenizer, buffered/streaming Unicode-trimmed string/list/no-system HTTP prompts, EOS/length/string stops, sampling rejection and native failure/reuse passed; scripted driver only",
  )
}
