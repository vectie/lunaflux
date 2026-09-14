///|
fn weight_view_request(
  family : @luna_projection_strategy.ProjectionSemanticFamily,
) -> ProjectionTileCompilerRequest raise {
  ProjectionTileCompilerRequest::new(
    dispatch=@luna_projection_strategy.plan_projection_dispatch(
      @luna_projection_strategy.ProjectionShape::new(
        family,
        input_width=128,
        output_width=256,
        intermediate_width=if family == GatedMlp { 512 } else { 0 },
      ),
      32,
      @luna_projection_strategy.ProjectionCapabilities::new(subgroup_width=32),
    ),
    maximum_rows=1,
    maximum_tokens=32,
    demand=AllTokenRows,
  )
}

///|
test "weight partitions survive normalization and distinguish semantic identity" {
  for family in [@luna_projection_strategy.ProjectionSemanticFamily::Dense, Qkv] {
    let request = weight_view_request(family)
    let unbound = compile_projection_tile(request)
    let first = compile_projection_tile(
      request.with_concatenated_weights([128, 64, 64]),
    )
    let second = compile_projection_tile(
      request.with_concatenated_weights([64, 96, 96]),
    )
    let four = compile_projection_tile(
      request.with_concatenated_weights([64, 64, 64, 64]),
    )
    assert_eq(unbound.program().concatenated_weights().length(), 0)
    assert_eq(
      first.program().concatenated_weights(),
      first.source_program().concatenated_weights(),
    )
    assert_eq(
      first.program().concatenated_weights(),
      plan_projection_concatenated_rows([128, 64, 64]).unwrap(),
    )
    assert_true(first.program().digest() != second.program().digest())
    assert_true(first.digest() != second.digest())
    assert_true(first.program().digest() != unbound.program().digest())
    assert_eq(four.program().concatenated_weights().length(), 4)
    assert_eq(
      first,
      compile_projection_tile(request.with_concatenated_weights([128, 64, 64])),
    )
  }
}

///|
test "weight bindings reject invalid partitions without mutating the request" {
  let request = weight_view_request(Dense)
  let before = compile_projection_tile(request)
  for
    counts in [[], [0, 256], [-1, 257], [128, 127], [128, 129], [2147483647, 1]] {
    try request.with_concatenated_weights(counts) catch {
      InvalidProjectionTileCompilation(InvalidProblem) => ()
      _ => fail("unexpected binding error")
    } noraise {
      _ => fail("invalid partition accepted")
    }
  }
  let mlp = weight_view_request(GatedMlp)
  try mlp.with_concatenated_weights([128, 128]) catch {
    InvalidProjectionTileCompilation(InvalidProblem) => ()
    _ => fail("unexpected role error")
  } noraise {
    _ => fail("multiple MLP weight roles were collapsed")
  }
  assert_eq(before, compile_projection_tile(request))
}

///|
test "ingress weight views cannot be rebound to different head partitions" {
  let request = weight_view_request(Qkv)
  // Use the public constructor: the epilogue establishes its own row view.
  let dispatch = @luna_projection_strategy.plan_projection_dispatch(
    @luna_projection_strategy.ProjectionShape::new(
      Qkv,
      input_width=128,
      output_width=256,
    ),
    32,
    @luna_projection_strategy.ProjectionCapabilities::new(subgroup_width=32),
  )
  let ingress = ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
    dispatch~,
    maximum_rows=1,
    maximum_tokens=32,
    epilogue=ProjectionAttentionIngressEpilogue::new(
      query_heads=2,
      kv_heads=1,
      head_dimension=64,
      norm_epsilon=0.000001,
      rope_theta=10000.0,
    ),
  )
  assert_eq(
    compile_projection_tile(ingress).program().concatenated_weights(),
    compile_projection_tile(request.with_concatenated_weights([128, 64, 64]))
    .program()
    .concatenated_weights(),
  )
  assert_eq(ingress, ingress.with_concatenated_weights([128, 64, 64]))
  try ingress.with_concatenated_weights([64, 96, 96]) catch {
    InvalidProjectionTileCompilation(InvalidProblem) => ()
    _ => fail("unexpected ingress binding error")
  } noraise {
    _ => fail("ingress partitions changed")
  }
}
