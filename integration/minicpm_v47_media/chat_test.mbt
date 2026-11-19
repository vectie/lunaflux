///|
async test "inline images normalize placeholders while retaining generation controls" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let png = @fs.read_file("media/rgb8/fixtures/rgb.png").binary()
  let jpeg = @fs.read_file("media/rgb8/fixtures/baseline.jpg").binary()
  let raw : Json = {
    "model": "MiniCPM-V-4.7-35B-A3B",
    "stream": false,
    "temperature": 0,
    "max_tokens": 8,
    "stop": ["done"],
    "messages": [
      { "role": "system", "content": "\u{00a0}policy\u{3000}" },
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "\u{000b}compare" },
          {
            "type": "image_url",
            "image_url": {
              "url": Json("data:image/png;base64," + @base64.encode(png[:])),
            },
          },
          { "type": "text", "text": "with" },
          {
            "type": "image_url",
            "image_url": {
              "url": Json("data:image/jpeg;base64," + @base64.encode(jpeg[:])),
              "detail": "high",
            },
          },
        ],
      },
    ],
  }
  let normalized = @minicpm_v47_media.normalize_chat(
    spec,
    @utf8.encode(raw.stringify()),
  )
  assert_true(
    try
      @minicpm_v47_media.normalize_chat(
        spec,
        @utf8.encode(raw.stringify()),
        allow_images=false,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(normalized.system_present())
  assert_eq(normalized.content().length(), 4)
  let first = @processor.Plan::build(spec, height=1, width=2)
  let second = @processor.Plan::build(spec, height=8, width=8)
  let expected = "compare\n" +
    first.placeholder_text(image_id=0) +
    "\nwith\n" +
    second.placeholder_text(image_id=1)
  assert_true(
    @json.parse(@utf8.decode(normalized.body()))
    is {
      "max_tokens": 8,
      "temperature": 0,
      "stop": ["done"],
      "messages": [
        { "role": "system", "content": "policy", .. },
        { "role": "user", "content": String(content), .. },
      ],
      ..
    } &&
    content == expected,
  )
}

///|
async test "optional system and text lists retain the pinned no-system template" {
  guard assets() is Some((spec, tokenizer, vision)) else { return }
  let normalized = @minicpm_v47_media.normalize_chat(
    spec,
    @utf8.encode(
      (
        {
          "model": "MiniCPM-V-4.7-35B-A3B",
          "messages": [{ "role": "user", "content": "  hello  " }],
        } : Json).stringify(),
    ),
  )
  assert_true(!normalized.system_present())
  let prepared = @minicpm_v47_media.prepare(
    spec,
    tokenizer,
    vision,
    normalized.system(),
    normalized.content()[:],
    maximum_tokens=4096,
    maximum_prepared_pixel_bytes=0L,
    system_present=normalized.system_present(),
  )
  assert_eq(
    prepared.prompt().tokens(),
    @prompt.prepare_text_prompt(
      spec,
      tokenizer,
      "",
      "hello",
      maximum_tokens=4096,
      system_present=false,
    ).tokens(),
  )
  assert_true(
    @prompt.render_text_prompt("", "hello", system_present=false).has_prefix(
      "<|im_start|>user\n",
    ),
  )
}

///|
async test "inline image orientation is resolved before model geometry" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let bytes = @fs.read_file("media/rgb8/fixtures/exif6.png").binary()
  let raw = @rgb8.Image::decode(
    bytes,
    maximum_encoded_bytes=4096,
    maximum_pixels=1000,
  )
  assert_eq((raw.width(), raw.height()), (17, 11))
  let request : Json = {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64," + @base64.encode(bytes[:]),
            },
          },
        ],
      },
    ],
  }
  let normalized = @minicpm_v47_media.normalize_chat(
    spec,
    @utf8.encode(request.stringify()),
  )
  guard normalized.content()[:] is [Image(image)] else {
    fail("inline image missing")
  }
  assert_eq((image.width(), image.height()), (11, 17))
  let expected = @rgb8.Image::decode_oriented(
    bytes,
    maximum_encoded_bytes=4096,
    maximum_pixels=1000,
  )
  assert_eq(image.data(), expected.data())
  let plan = @processor.Plan::build(spec, width=11, height=17)
  assert_true(
    @json.parse(@utf8.decode(normalized.body()))
    is { "messages": [{ "content": String(text), .. }], .. } &&
    text == plan.placeholder_text(image_id=0),
  )
}

///|
async test "normalization rejects duplicate keys invalid data and unsupported media instead of fetching" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let sources = [
    "{\"model\":\"a\",\"model\":\"b\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
    "{\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"cont\\u0065nt\":\"y\"}]}",
  ]
  for text in sources {
    assert_true(
      (try? @minicpm_v47_media.normalize_chat(spec, @utf8.encode(text)))
      is Err(_),
    )
  }
  let png = @fs.read_file("media/rgb8/fixtures/rgb.png").binary()
  for
    url in [
      "https://example.invalid/image.png",
      "file:///image.png",
      "data:image/png;base64,!",
      "data:image/jpeg;base64," + @base64.encode(png[:]),
    ] {
    let raw : Json = {
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "image_url", "image_url": { "url": Json(url) } },
          ],
        },
      ],
    }
    assert_true(
      (try? @minicpm_v47_media.normalize_chat(
        spec,
        @utf8.encode(raw.stringify()),
      ))
      is Err(_),
    )
  }
}
