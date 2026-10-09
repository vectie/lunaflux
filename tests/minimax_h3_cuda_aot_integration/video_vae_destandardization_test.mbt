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
      #|  "df9ed95b3e72c427f57903293ab5cf4dca84e4724fab04738c1f7a6403a77f70",
      #|  "964e610ceffb91764687f86948228f796a16d50a7eb60b566ab5f6bffe2e0db9",
      #|]
    ),
  )
  debug_inspect(
    parameter_digests,
    content=(
      #|[
      #|  "08b469f15bf084e21690c23e472ba1f5afd3a43aa99e7e6219e8e8573a1c6820",
      #|  "ee8b73a877cc1597bfdeb6f02bf4bc30c0415da672b83bb57673b72f4fa0153c",
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
      #|  "f72e6ffa3e36bb1b3ea6737b38b484c52880797eb34135b515a8e101c690712c",
      #|  "664151b53a4ca3fb6366b8a0bfa3299477c200537ec356277a8175930d1cf5ac",
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
