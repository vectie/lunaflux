///|
async fn run_media_queue_fixture(
  spec : @model.Spec,
  vision : @vision.Program,
  execution : @image_runtime.Execution,
) -> Unit {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(assets) else {
    println("SKIP actual media queue ABI: set LUNA_MINICPM_V47_ASSET_DIR")
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
  let root = @approved_fs.ApprovedRoot::open_absolute(
    @fs.realpath("media/rgb8/fixtures"),
  )
  defer root.close()
  let queue = @media.prepare_requests(
    spec,
    tokenizer,
    vision,
    @fs.read_file("cmd/minicpm_v47_generate/media-requests.example.json").json(),
    load_image=path => {
      let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(path))
      defer file.close()
      @rgb8.Image::decode(
        file.read_immutable_snapshot(maximum_bytes=4096L),
        maximum_encoded_bytes=4096,
        maximum_pixels=64,
      )
    },
    maximum_prepared_pixel_bytes=10000000L,
  )
  assert_eq(queue.length(), 2)
  assert_eq(queue[0].input.image_count(), 2)
  assert_eq(queue[0].input.patch_rows(), 2080)
  assert_eq(queue[0].input.feature_rows(), 130)
  assert_eq(queue[1].input.image_count(), 0)
  for request in queue {
    match request.input.vision_input() {
      Some((prompt, input)) => {
        let before = reads()
        execution.begin_image(
          prompt,
          input,
          maximum_new_tokens=request.maximum_new_tokens,
          stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
        )
        execution.wait()
        assert_eq(reads(), before)
      }
      None =>
        execution.begin_text(
          request.input.prompt(),
          maximum_new_tokens=request.maximum_new_tokens,
          stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
        )
    }
    let chunks = (request.input.prompt().tokens().length() + 31) / 32
    set_sample_token(248046)
    for index in 0..<chunks {
      measured_image_request_step(execution, produces_token=index + 1 == chunks)
    }
    assert_true(execution.is_finished())
    assert_eq(execution.emitted_tokens(), [])
    assert_eq(
      execution.finish_reason(),
      Some(@inference.FinishReason::StopSequence),
    )
  }
  println(
    "MiniCPM-V 4.7 media ABI: PNG/JPEG -> pinned tokenizer -> multi-image processor/layout -> vision -> decoder/EOS -> text reuse passed; scripted driver only",
  )
}
