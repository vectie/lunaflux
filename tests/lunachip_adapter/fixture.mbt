///|
pub fn contract_text() -> String {
  let value : Json = {
    "schema": "lunachip.pblaze3-device/v1",
    "target": "pblaze3-2gib",
    "control_abi": 65537,
    "device_id": 1346515760,
    "ddr_bytes": "2147483648",
    "descriptor_words": 16,
    "program_descriptors": 2048,
    "scratch_words": 32768,
    "vector_limit": 4096,
    "matrix_k_multiple": 64,
    "matrix_alignment_bytes": 128,
    "numeric_profile": "qwen-layer-ordered-fp32-approx-v1",
    "weight_layout": "bf16-row-major-or-packed16-v1",
  }
  value.stringify()
}

///|
pub fn metadata() -> @qwen.Qwen3ModelMetadata raise {
  @qwen.Qwen3ModelMetadata::from_verified_content(
    @qwen.Qwen3ModelSpec::new(
      vocabulary_size=151936,
      hidden_size=1024,
      intermediate_size=3072,
      layer_count=28,
      attention_head_count=16,
      kv_head_count=8,
      head_dimension=128,
      context_length=40960,
      rope_theta=1000000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
      tied_embeddings=true,
    ),
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn id(value : Int32) -> @tile.LunaTileTensorId raise {
  @tile.LunaTileTensorId::from_stable_int(value)
}

///|
fn region(value : Int32) -> @tile.LunaTileRegion raise {
  @tile.LunaTileRegion::new(
    tensor=id(value),
    origins=[@tile.LunaTileAffineExpr::new(constant=0L, terms=[])],
    extents=[4L],
  )
}

///|
pub fn program(
  dtype~ : @tile.LunaTileDType = Float32,
  initialized~ : Bool = true,
  kind~ : @tile.LunaTileElementwiseKind = Add,
  loops~ : ArrayView[Int64] = [1L],
) -> @tile.LunaTileProgram raise {
  let tensors = []
  for i in 1..<=6 {
    tensors.push(
      @tile.LunaTileTensorView::new(
        id=id(i),
        dtype~,
        address_space=if i <= 3 { Global } else { Register },
        shape=[4L],
        strides_bytes=[if dtype == BFloat16 { 2L } else { 4L }],
        offset_bytes=0L,
        alignment_bytes=4L,
      ),
    )
  }
  let ops : Array[@tile.LunaTileInstruction] = []
  if initialized {
    ops.push(LunaTileCopy(region(1), region(4)))
    ops.push(LunaTileCopy(region(2), region(5)))
  }
  ops.push(@tile.LunaTileInstruction::elementwise(kind, id(6), [id(4), id(5)]))
  ops.push(
    @tile.LunaTileInstruction::elementwise(Multiply, id(6), [id(6), id(5)]),
  )
  ops.push(LunaTileCopy(region(6), region(3)))
  @tile.admit_luna_tile_program(
    @tile.LunaTileProgramInput::new(
      loop_extents=loops,
      tensors~,
      instructions=ops,
      constraints=[],
    ),
    @tile.LunaTileLimits::new(
      max_tensors=16,
      max_instructions=32,
      max_rank=2,
      max_loop_rank=2,
      max_constraints=8,
      max_canonical_bytes=32768,
      max_shared_memory_bytes=0L,
    ),
  )
}

///|
pub fn bindings() -> Array[@compiler.GlobalBinding] raise {
  [
    { tensor: id(1), address: 0x7ffffff0UL, },
    { tensor: id(2), address: 0x50000000UL, },
    { tensor: id(3), address: 0x40000004UL, },
  ]
}

///|
/// Generate the actual cross-repository test payload. The caller supplies the
/// independently exported LunaChip contract, exercising real schema agreement.
pub fn export_fixture(contract : String) -> String raise {
  let tile = program()
  let lowered = @compiler.lower(
    @device.parse_contract(contract),
    tile,
    bindings(),
  )
  assert_eq(lowered.chunk_count(), 1)
  let bytes = lowered.chunk(0).unwrap()
  let words = Array::makei(bytes.length() / 4, i => {
    bytes[i * 4].to_uint() |
    (bytes[i * 4 + 1].to_uint() << 8) |
    (bytes[i * 4 + 2].to_uint() << 16) |
    (bytes[i * 4 + 3].to_uint() << 24)
  })
  let left = [1.0F, -2.0F, 0.5F, 3.25F]
  let right = [2.0F, 4.0F, -0.25F, -1.0F]
  let output = Array::makei(4, i => {
    let sum = left[i] + right[i]
    (sum * right[i]).reinterpret_as_uint()
  })
  let fixture : Json = {
    "schema": "lunachip.adapter-fixture/v1",
    "source_program": tile.digest().as_hex(),
    "program_words": @json.to_json(words),
    "assets": [
      {
        "address": "2147483632",
        "words": @json.to_json(left.map(x => x.reinterpret_as_uint())),
      },
      {
        "address": "1342177280",
        "words": @json.to_json(right.map(x => x.reinterpret_as_uint())),
      },
    ],
    "outputs": [{ "address": "1073741828", "words": @json.to_json(output) }],
  }
  fixture.stringify(indent=2) + "\n"
}
