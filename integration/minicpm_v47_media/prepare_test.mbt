///|
async fn assets() -> (@model.Spec, @tokenizer.TokenizerSpec, @vision.Program)? {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(root) else {
    println("SKIP actual media tokenizer: set LUNA_MINICPM_V47_ASSET_DIR")
    return None
  }
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let tokenizer = @prompt.prepare_tokenizer(
    spec,
    @fs.read_file(root + "/tokenizer.json").binary()[:],
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
  Some(
    (
      spec,
      tokenizer,
      @vision.Program::new(
        spec,
        maximum_patches=2048,
        maximum_crop_patches=1024,
      ),
    ),
  )
}

///|
async test "decoded camera frames share one prompt layout and pixel preparation" {
  guard assets() is Some((spec, tokenizer, vision)) else { return }
  let red = @rgb8.Image::from_rgb(
    Bytes::from_array(
      Array::makei(8 * 8 * 3, i => if i % 3 == 0 { 255 } else { 0 }),
    ),
    width=8,
    height=8,
    maximum_pixels=64,
  )
  let green = @rgb8.Image::from_rgb(
    Bytes::from_array(
      Array::makei(8 * 8 * 3, i => if i % 3 == 1 { 255 } else { 0 }),
    ),
    width=8,
    height=8,
    maximum_pixels=64,
  )
  let prepared = @minicpm_v47_media.prepare(
    spec,
    tokenizer,
    vision,
    "policy",
    [Text("compare"), Image(red), Text("with"), Image(green)],
    maximum_tokens=4096,
    maximum_prepared_pixel_bytes=2408448L,
  )
  assert_eq(prepared.image_count(), 2)
  assert_eq(prepared.pixel_bytes(), 2408448L)
  assert_eq(prepared.patch_rows(), 2048)
  assert_eq(prepared.feature_rows(), 128)
  let plan = @processor.Plan::build(spec, height=8, width=8)
  assert_eq(plan.preparation_bytes(), 2419456L)
  let expected = @multimodal.prepare_image_prompt(
    spec,
    tokenizer,
    "policy",
    [Text("compare"), Image(plan), Text("with"), Image(plan)],
    maximum_tokens=4096,
  )
  assert_eq(prepared.prompt().tokens(), expected.prompt().tokens())
  guard prepared.vision_input() is Some((prompt, input)) else {
    fail("missing image execution input")
  }
  assert_eq(prompt.feature_rows(), input.feature_rows())
  assert_true(
    try
      @minicpm_v47_media.prepare(
        spec,
        tokenizer,
        vision,
        "policy",
        [Image(red), Image(green)],
        maximum_tokens=4096,
        maximum_prepared_pixel_bytes=2408447L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @minicpm_v47_media.prepare(
        spec,
        tokenizer,
        vision,
        "policy",
        [Image(red), Image(green), Image(red)],
        maximum_tokens=4096,
        maximum_prepared_pixel_bytes=10000000L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @minicpm_v47_media.prepare(
        spec,
        tokenizer,
        vision,
        "policy",
        [Image(red)],
        maximum_tokens=4096,
        maximum_prepared_pixel_bytes=1204224L,
        maximum_image_working_bytes=2419455L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}

///|
async test "text content list uses the pinned newline and trim law" {
  guard assets() is Some((spec, tokenizer, vision)) else { return }
  let prepared = @minicpm_v47_media.prepare(
    spec,
    tokenizer,
    vision,
    " policy ",
    [Text("  one"), Text("two  ")],
    maximum_tokens=4096,
    maximum_prepared_pixel_bytes=0L,
  )
  assert_eq(prepared.image_count(), 0)
  assert_eq(prepared.pixel_bytes(), 0L)
  assert_eq(prepared.patch_rows(), 0)
  assert_eq(prepared.feature_rows(), 0)
  assert_true(prepared.vision_input() is None)
  assert_eq(
    prepared.prompt().tokens(),
    @prompt.prepare_text_prompt(
      spec,
      tokenizer,
      "policy",
      "one\ntwo",
      maximum_tokens=4096,
    ).tokens(),
  )
}

///|
async test "media queue resolves image occurrences in order and admits aggregate packed bytes" {
  guard assets() is Some((spec, tokenizer, vision)) else { return }
  let image = @rgb8.Image::decode(
    @fs.read_file("media/rgb8/fixtures/baseline.jpg").binary(),
    maximum_encoded_bytes=4096,
    maximum_pixels=64,
  )
  let seen : Array[String] = []
  let document : Json = {
    "format": "lunaflux.minicpm-v47-media-requests.v1",
    "requests": [
      {
        "id": "a",
        "system": "policy",
        "maximum_new_tokens": 1,
        "user": [
          { "type": "image", "path": "first.jpg" },
          { "type": "text", "text": "compare" },
          { "type": "image", "path": "second.jpg" },
        ],
      },
    ],
  }
  let queue = @minicpm_v47_media.prepare_requests(
    spec,
    tokenizer,
    vision,
    document,
    load_image=path => {
      seen.push(path)
      image
    },
    maximum_prepared_pixel_bytes=2408448L,
  )
  assert_eq(seen, ["first.jpg", "second.jpg"])
  assert_eq(queue.length(), 1)
  assert_eq(queue[0].input.image_count(), 2)
  assert_eq(queue[0].maximum_new_tokens, 1)
  assert_eq(queue[0].context_ceiling, 4096)
  assert_true(
    try
      @minicpm_v47_media.prepare_requests(
        spec,
        tokenizer,
        vision,
        document,
        load_image=_ => image,
        maximum_prepared_pixel_bytes=2408447L,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let bad : Array[Json] = [
    Json::null(),
    { "format": "lunaflux.minicpm-v47-media-requests.v1", "requests": [] },
    {
      "format": "lunaflux.minicpm-v47-media-requests.v1",
      "requests": [
        {
          "id": "a",
          "system": "p",
          "maximum_new_tokens": 1,
          "temperature": 0.5,
          "user": [{ "type": "text", "text": "a" }],
        },
      ],
    },
    {
      "format": "lunaflux.minicpm-v47-media-requests.v1",
      "requests": [
        {
          "id": "a",
          "system": "p",
          "maximum_new_tokens": 1,
          "user": [{ "type": "image_url", "path": "remote" }],
        },
      ],
    },
  ]
  for invalid in bad {
    assert_true(
      try
        @minicpm_v47_media.prepare_requests(
          spec,
          tokenizer,
          vision,
          invalid,
          load_image=_ => image,
          maximum_prepared_pixel_bytes=2408448L,
        )
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
}
