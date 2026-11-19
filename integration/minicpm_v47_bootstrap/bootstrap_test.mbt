///|
async test "shared Spark program retains the full flagship decoder and initial AOT geometry" {
  let model = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(model, maximum_rows=32)
  let attention = @minicpm_v47_bootstrap.compile_spark_attention(operators)
  let program = @minicpm_v47_bootstrap.text_program(operators, attention)
  assert_eq(program.decoder_program().layers().length(), 40)
  assert_eq(program.launch_count(Prefill), 924)
  assert_eq(program.text_source().plan().geometry(), (248144, 2048, 1, 32, 1))
  assert_eq(
    program.decoder_program().attention_program().context_tokens(),
    4096,
  )
}

///|
async fn fixture_model(directory : String, assets : String) -> Json {
  for name in ["config.json", "tokenizer.json"] {
    @fs.write_file(
      directory + "/" + name,
      @fs.read_file(assets + "/" + name),
      create_mode=CreateNew,
    )
  }
  let files = @fs.opendir("model/minicpm_v47_weights/fixtures")
  defer files.close()
  let names = files.read_all()
  names.sort()
  let index : Map[String, Json] = Map([])
  for filename in names {
    let name = filename.strip_suffix(".header.json").unwrap().to_owned()
    let header = @fs.read_file("model/minicpm_v47_weights/fixtures/" + filename).binary()
    guard @json.parse(@utf8.decode(header)) is Object(tree) else {
      fail("header fixture")
    }
    let mut end = 0L
    for tensor, entry in tree {
      if tensor == "__metadata__" {
        continue
      }
      guard entry is { "data_offsets": [_, Number(last, ..)], .. } else {
        fail("tensor extent")
      }
      end = end.max(last.to_int64())
      index[tensor] = Json(name)
    }
    let prefix = @buffer.Buffer()
    prefix.write_uint64_le(header.length().to_uint64())
    prefix.write_bytes(header)
    let path = directory + "/" + name
    @fs.write_file(path, prefix.to_bytes(), create_mode=CreateNew)
    let file = @fs.open(path, mode=WriteOnly)
    defer file.close()
    file.write_at(
      b"\x00"[:],
      position=8L + header.length().to_int64() + end - 1L,
    )
  }
  let document : Json = { "weight_map": Json(index) }
  @fs.write_file(
    directory + "/model.safetensors.index.json",
    @utf8.encode(document.stringify()),
    create_mode=CreateNew,
  )
  document
}

///|
async test "actual config tokenizer and sparse headers prepare flagship requests without CUDA" {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(assets) else {
    println("SKIP actual bootstrap assets: set LUNA_MINICPM_V47_ASSET_DIR")
    return
  }
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-v47-bootstrap"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let index = fixture_model(directory, assets)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let image_model = @minicpm_v47_bootstrap.prepare_images(
      root,
      maximum_patches=16384,
      maximum_crop_patches=8192,
    )
    assert_eq(image_model.vision_weights.entries().length(), 459)
    assert_eq(image_model.vision_weights.byte_count(), 1105972064L)
    assert_eq(image_model.vision.stages().length(), 318)
    let prepared = image_model.text
    assert_eq(
      (prepared.inspected_tensors, prepared.inspected_shards),
      (1152, 16),
    )
    assert_eq(prepared.weights.allocations().length(), 653)
    assert_eq(prepared.weights.byte_count(), 69319779584L)
    assert_eq(prepared.program.launch_count(Prefill), 924)
    assert_eq(prepared.program.launch_count(Decode), 924)
    assert_eq(
      prepared.program.decoder_program().attention_program().context_tokens(),
      4096,
    )
    let requests = @minicpm_v47_bootstrap.prepare_requests(
      prepared.model,
      prepared.tokenizer,
      {
        "format": "lunaflux.minicpm-v47-text-requests.v1",
        "requests": [
          {
            "id": "one",
            "system": "Select a skill.",
            "user": "向前走两步。",
            "maximum_new_tokens": 64,
          },
          {
            "id": "two",
            "system": "Select a skill.",
            "user": "Stand still.",
            "maximum_new_tokens": 1,
            "context_ceiling": 256,
          },
        ],
      },
    )
    assert_eq(requests.length(), 2)
    assert_eq(
      (requests[0].maximum_new_tokens, requests[0].context_ceiling),
      (64, 4096),
    )
    assert_eq(
      (requests[1].maximum_new_tokens, requests[1].context_ceiling),
      (1, 256),
    )
    assert_eq(requests[0].prompt.tokens()[0], 248045)
    assert_eq(
      prepared.tokenizer.decode_text(
        requests[0].prompt.tokens(),
        special_policy=Preserve,
      ),
      @prompt.render_text_prompt("Select a skill.", "向前走两步。"),
    )
    assert_eq(requests[0].prompt.positions().rope_delta(), 0)
    let invalid : Array[Json] = [
      Json::null(),
      { "weight_map": {} },
      { "weight_map": { "a": 1 } },
    ]
    for document in invalid {
      assert_true(
        try @minicpm_v47_bootstrap.inspect_checkpoint(root, document) catch {
          _ => true
        } noraise {
          _ => false
        },
      )
    }
    guard index is { "weight_map": Object(owners), .. } else {
      fail("fixture index")
    }
    let wrong = owners.copy()
    let entries = owners.to_array()
    let first = entries[0]
    let different = entries.filter(entry => entry.1 != first.1)[0]
    wrong[first.0] = different.1
    assert_true(
      try
        @minicpm_v47_bootstrap.inspect_checkpoint(root, {
          "weight_map": Json(wrong),
        })
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    assert_true(
      try
        @minicpm_v47_bootstrap.read_asset(root, "config.json", maximum_bytes=4L)
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
}

///|
async test "explicit greedy queue rejects unsupported media sampling and invalid token limits" {
  guard @env.get_env_var("LUNA_MINICPM_V47_ASSET_DIR") is Some(assets) else {
    println("SKIP actual request assets: set LUNA_MINICPM_V47_ASSET_DIR")
    return
  }
  let model = @model.Spec::parse(
    @fs.read_file(assets + "/config.json").binary()[:],
  )
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
  let base : Map[String, Json] = Map([
    ("id", "one"),
    ("system", "Policy"),
    ("user", "Stand"),
    ("maximum_new_tokens", 1),
  ])
  let invalid = [
    ("temperature", Json(0.5)),
    ("images", Json::null()),
    ("maximum_new_tokens", Json(0)),
    ("maximum_new_tokens", Json(1.5)),
    ("maximum_new_tokens", Json(4097)),
    ("context_ceiling", Json(1)),
  ]
  for (field, value) in invalid {
    let fields = base.copy()
    fields[field] = value
    assert_true(
      try
        @minicpm_v47_bootstrap.prepare_requests(model, tokenizer, {
          "format": "lunaflux.minicpm-v47-text-requests.v1",
          "requests": [Json(fields)],
        })
      catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
  assert_true(
    try
      @minicpm_v47_bootstrap.prepare_requests(model, tokenizer, {
        "format": "lunaflux.minicpm-v47-text-requests.v1",
        "requests": [Json(base), Json(base)],
      })
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try
      @minicpm_v47_bootstrap.prepare_requests(model, tokenizer, {
        "format": "lunaflux.minicpm-v47-text-requests.v1",
        "requests": [],
      })
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
