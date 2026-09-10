///|
test "bounded pipeline specialization is exported without inventing autotune results" {
  for
    parameters in [
      ProjectionParameters::{
        family: QkvProjection,
        input_width: 1024,
        output_width: 4096,
        query_width: 2048,
        intermediate_width: 1024,
      },
      ProjectionParameters::{
        family: DenseProjection,
        input_width: 2048,
        output_width: 1024,
        query_width: 0,
        intermediate_width: 0,
      },
      ProjectionParameters::{
        family: GatedMlp,
        input_width: 1024,
        output_width: 1024,
        query_width: 0,
        intermediate_width: 3072,
      },
    ] {
    let runtime = projection_build_runtime_dispatch(
      parameters,
      1024,
      projection_fixture_target(),
      [],
    )
    let primary = runtime.dispatch_plan(1024)
    assert_eq(runtime.select(16).source(), StaticFallback)
    let (source, variants) = projection_row_variant_sources(
      parameters,
      row_variant_test_profile(),
      runtime,
      primary,
      "bounded",
      max_registers=128,
    )
    let expected = if parameters.family == GatedMlp {
      [8, 16, 32]
    } else {
      [8, 16]
    }
    assert_eq(variants.length(), expected.length())
    for i, variant in variants {
      let bound = expected[i]
      assert_eq(variant.maximum_token_rows(), bound)
      assert_eq(variant.function_symbol(), "bounded_rows\{bound}")
      assert_eq(
        variant.dimensions().block_x(),
        primary.prefill().output_tiles_per_workgroup() * 32,
      )
      assert_true(source.contains("token_count > \{bound}"))
      assert_eq(
        variant.companion_dimensions() is Some(_),
        parameters.family == GatedMlp,
      )
    }
    // No redundant full-envelope entry, and the semantic row1 branch survives.
    assert_false(source.contains("void bounded_rows1024("))
    assert_true(source.contains("token_count == 1"))
    assert_eq(runtime.select(16).source(), StaticFallback)
  }
}
