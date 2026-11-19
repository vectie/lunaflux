///|
#borrow(tokens)
extern "C" fn set_sample_script(tokens : FixedArray[Int32]) -> Bool = "lunaflux_device_worker_fake_set_sample_script"

///|
extern "C" fn sample_script_done() -> Bool = "lunaflux_device_worker_fake_sample_script_done"

///|
/// One-request loopback fixture for the real MoonRobo planner client. It uses
/// the full selected program and real tokenizer, with explicitly scripted
/// sampled IDs and borrowed synthetic weights. It performs no GPU arithmetic.
async fn serve_planner_fixture(
  spec : @model.Spec,
  text : @text.Program,
  context : @device.Context,
  layers : ArrayView[@decoder.LayerWeights],
  text_weights : @text.Weights,
  ready : String,
  proposal : String,
) -> Unit {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(assets) else {
    fail("planner fixture requires actual tokenizer assets")
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
  let reply = tokenizer.encode_text(
      @fs.read_file(proposal).text().trim().to_owned(),
      special_policy=AsOrdinary,
      output_limit=255,
      truncation=RejectOverflow,
    ).token_ids
  let tokens = FixedArray::makei(reply.length() + 1, i => {
    if i < reply.length() {
      reply[i]
    } else {
      248046
    }
  })
  assert_true(set_sample_script(tokens))
  let shared = context.allocate(158662656L)
  defer shared.close()
  let vision = @vision.Program::new(
    spec,
    maximum_patches=4096,
    maximum_crop_patches=2048,
  )
  let weights = @vision.Weights::borrow(spec, vision, context, fn(_) {
    return shared
  })
  let execution = @image_runtime.Execution::new(
    @multimodal.Program::new(text),
    vision,
  )
  defer execution.close()
  execution.prepare(
    context,
    {
      text: {
        envelope: b"fake",
        recurrent: b"fake",
        moe: b"fake",
        attention_common: b"fake",
        attention_prefill: b"fake",
        attention_decode: b"fake",
        text: b"fake",
      },
      vision: b"fake",
      image_scatter: b"fake",
    },
    layers~,
    text_weights~,
    vision_weights=weights,
    maximum_mutable_bytes=@image_runtime.Budget::new(
        @multimodal.Program::new(text),
        vision,
        text_weight_bytes=0L,
        vision_weight_bytes=0L,
      ).mutable_bytes,
  )
  let backend = @chat.Backend::with_images(spec, tokenizer, vision, execution)
  let key = @utf8.encode("software-fixture-key").to_fixedarray()
  let policy = @auth.LunaApiAuthPolicy::new(
    key,
    source_offset=0,
    length=key.length(),
    maximum_credential_bytes=64,
  )
  defer policy.close()
  let service = backend.service(policy, timeout_ms=10000)
  defer service.close()
  let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
  defer listener.close()
  @async.with_timeout(20000, () => {
    @async.with_task_group() <| group => {
      group.spawn_bg(no_wait=true, () => service.serve(listener))
      let ready_staging = ready + ".tmp"
      @fs.write_file(
        ready_staging,
        @utf8.encode(
          (
            {
              "endpoint": Json("http://127.0.0.1:\{listener.addr.port()}"),
              "model": "MiniCPM-V-4.7-35B-A3B",
              "scope": "scripted full-program software fixture; no GPU computation",
            } : Json).stringify(),
        ),
        create_mode=CreateNew,
      )
      @fs.rename(ready_staging, ready)
      while !sample_script_done() || !service.is_ready() {
        @async.pause()
      }
      assert_true(execution.is_finished())
      group.return_immediately(())
    }
  })
}
