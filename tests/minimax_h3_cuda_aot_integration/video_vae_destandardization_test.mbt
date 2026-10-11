///|
fn video_destandardization_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      VideoVaeLatentDestandardization(_, _, _) => return ordinal
      _ => ()
    }
  }
  abort("official video latent destandardization missing")
}

///|
test "FL2VA and Ref2VA bind exact official video latent destandardization" {
  let requirement_digests : Array[String] = []
  let parameter_digests : Array[String] = []
  let source_digests : Array[String] = []
  let recipe_digests : Array[String] = []
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  for variant in variants {
    let requirements = official_requirements(variant)
    let request = try! requirements
      .shape_contract()
      .resolve_request(
        height=768,
        width=1344,
        requested_frames=120,
        inference_steps=50,
      )
    let parameters = official_video_vae_destandardization(variant, request)
    assert_eq(parameters.batch_count(), 1)
    assert_eq(parameters.channel_count(), 24)
    assert_eq(parameters.elements_per_channel(), 149184L)
    assert_eq(parameters.element_count(), 3580416L)
    assert_eq(
      parameters.mean(0).unwrap().reinterpret_as_uint(),
      Float::from_double(0.858090341091156).reinterpret_as_uint(),
    )
    assert_eq(
      parameters.standard_deviation(23).unwrap().reinterpret_as_uint(),
      Float::from_double(2.6127843856811523).reinterpret_as_uint(),
    )
    let candidate = try! @joint_diffusion_cuda_aot.lower_video_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=video_destandardization_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(candidate.batch_count(), 1)
    assert_eq(candidate.channel_count(), 24)
    assert_eq(candidate.elements_per_channel(), 149184L)
    assert_eq(candidate.element_count(), 3580416L)
    assert_eq(candidate.operands()[0].byte_count(), 14321664L)
    assert_eq(candidate.operands()[1].byte_count(), 96L)
    assert_eq(candidate.operands()[2].byte_count(), 96L)
    assert_eq(candidate.operands()[3].byte_count(), 14321664L)
    assert_eq(candidate.dimensions().grid_x(), 13986)
    assert_false(candidate.manifest_bindable())
    requirement_digests.push(requirements.digest().as_hex())
    parameter_digests.push(parameters.digest().as_hex())
    source_digests.push(candidate.source_digest().as_hex())
    recipe_digests.push(candidate.recipe_digest().as_hex())
  }
  debug_inspect(
    requirement_digests,
    content=(
      #|[
      #|  "07693309235e6bd6318689a3bbed0c12414656b62143b8b8b9df1fd813f78972",
      #|  "f296b7089e950c42a19502e0e32aaaf39b07181c39350d8fd93a743a460dcecc",
      #|]
    ),
  )
  debug_inspect(
    parameter_digests,
    content=(
      #|[
      #|  "7052fe60e591d7d40b1d3fe4e3a005c851575f6c2eebf18f6b0f8d9c88884355",
      #|  "3bccc990948185dfb057523c2f200f60181de155ecf2c7c027081234d261b112",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "dad9050c8ceed0a0aed2deb80ca14a2ec5dfffc398b1dc97c0eb7000792d6320",
      #|  "dad9050c8ceed0a0aed2deb80ca14a2ec5dfffc398b1dc97c0eb7000792d6320",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "8ee07d604addeb8eaeedf88598a5975c090774c188dcc2decf2504d3ef4e84db",
      #|  "874df3872e0fc4769ec6acb6535a086311e92d2f26a8fab8cf2167c27a8aff50",
      #|]
    ),
  )
}

///|
test "official profile retains ordered decoder body after exact prefix" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  for variant in variants {
    let official = official_requirements(variant)
    let requirements = official.requirements()
    let ordinal = video_destandardization_ordinal(official)
    assert_eq(
      requirements[ordinal].operation(),
      VideoVaeLatentDestandardization(
        24,
        Float32InputScaleMeanOutput,
        BatchChannelFrameHeightWidth,
      ),
    )
    assert_eq(
      requirements[ordinal + 1].operation(),
      VideoVaeDecoderBody(24, 16, 17, 5),
    )
    assert_eq(requirements[ordinal + 1].capability(), VideoVaeDecoderBodyAot)
  }
}
