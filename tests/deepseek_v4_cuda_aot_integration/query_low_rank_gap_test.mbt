///|
test "all official profiles split query A norm and query B geometry exactly" {
  let cases = [
    ("deepseek-ai/DeepSeek-V4-Flash", '1', 1024, 64),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 1024, 64),
    ("deepseek-ai/DeepSeek-V4-Pro", '3', 1536, 128),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 1536, 128),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 1024, 64),
  ]
  for entry in cases {
    let (model_id, fill, rank, heads) = entry
    let requirements = official_deepseek_requirements(model_id, fill)
    assert_eq(
      requirements.requirements()[1].operation(),
      QueryLowRankAProjection(rank),
    )
    assert_eq(
      requirements.requirements()[1].capability(),
      QueryLowRankAProjectionAot,
    )
    assert_eq(
      requirements.requirements()[3].operation(),
      QueryKeyRmsNorm(0.000001, rank, 512),
    )
    assert_eq(
      requirements.requirements()[4].operation(),
      QueryLowRankBProjection(rank, heads, 512),
    )
    assert_eq(
      requirements.requirements()[4].capability(),
      QueryLowRankBProjectionAot,
    )
  }
}

///|
test "BF16 dense and LM-head lowerers cannot impersonate FP8 block query stages" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", 'a',
  )
  for ordinal in [1, 4] {
    try
      @advanced_decoder_cuda_aot.lower_language_model_head_cuda_aot_candidate(
        requirements,
        requirement_ordinal=ordinal,
        maximum_tokens=2,
        target=official_deepseek_target(),
        compiler=official_deepseek_compiler(),
        limits=official_deepseek_limits(),
      )
    catch {
      UnsupportedRequirement(actual) => assert_eq(actual, ordinal)
      _ => fail("wrong query-stage rejection")
    } noraise {
      _ => fail("an FP8 block-scaled query projection used a BF16 dense ABI")
    }
  }
}

///|
test "all official profiles lower replicated Query-A with exact raw ABI" {
  for
    entry in [
      ("deepseek-ai/DeepSeek-V4-Flash", '1', 4096, 1024, 4194304L, 256L),
      ("deepseek-ai/DeepSeek-V4-Flash-Base", '2', 4096, 1024, 4194304L, 256L),
      ("deepseek-ai/DeepSeek-V4-Pro", '3', 7168, 1536, 11010048L, 672L),
      ("deepseek-ai/DeepSeek-V4-Pro-Base", '4', 7168, 1536, 11010048L, 672L),
      ("deepseek-ai/DeepSeek-V4-Flash-0731", '5', 4096, 1024, 4194304L, 256L),
    ] {
    let requirements = official_deepseek_requirements(entry.0, entry.1)
    let candidate = @advanced_decoder_cuda_aot.lower_query_a_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=1,
      maximum_tokens=2,
      target=official_deepseek_target(),
      compiler=official_deepseek_compiler(),
      limits=official_deepseek_limits(),
    )
    assert_eq(candidate.input_width(), entry.2)
    assert_eq(candidate.output_width(), entry.3)
    assert_eq(candidate.operands().map(op => op.byte_count()), [
      20L,
      2L * entry.2.to_int64() * 2L,
      entry.4,
      entry.5,
      2L * entry.3.to_int64() * 2L,
    ])
    assert_true(candidate.tensor_parallel_replicated())
    assert_false(candidate.manifest_bindable())
    if entry.0 == "deepseek-ai/DeepSeek-V4-Flash-0731" {
      inspect(
        candidate.source_digest().as_hex(),
        content="86837864be2271db200acd3d24085684e5b737b8a99739c88941fb319c602816",
      )
    } else if entry.0 == "deepseek-ai/DeepSeek-V4-Pro" {
      inspect(
        candidate.source_digest().as_hex(),
        content="1bf8871c544c00a81da835113ba606e2628fed883d73223c3e7f0f0651097d2a",
      )
    }
  }
}
