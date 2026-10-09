///|
fn routing_fixture(
  format : @format.Format,
  score : @precision.MoeRoutingScore,
  selection? : @precision.MoeRoutingSelection = GroupedTopK,
) -> @precision.MoeRoutingPrecision raise {
  let representation = @format.Representation::new(format~, rows=4, columns=32)
  @precision.MoeRoutingPrecision::new(
    weight=@precision.plan_operand(
      role=Weight,
      representation~,
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
      memory_ceiling=representation.payload_bytes(),
    ),
    maximum_rows=2,
    group_count=2,
    group_score_topk=1,
    selected_group_count=1,
    selected_experts=2,
    score~,
    selection~,
    normalize=true,
    scaling=2.5,
    maximum_workspace_bytes=128L,
  )
}

///|
test "numeric routing lowers both weight formats and score operations" {
  for
    (format, score, symbol) in [
      (@format.Format::Bf16, @precision.MoeRoutingScore::Sigmoid, "route_bf16"),
      (F32, SqrtSoftplus, "route_f32"),
    ] {
    let source = @moe_routing_cuda_source.routing_source(
      routing_fixture(format, score),
      symbol~,
    )
    assert_eq(
      source.symbols()[:],
      [symbol + "_project", symbol + "_score", symbol + "_choose"][:],
    )
    assert_eq(source.launches()[:], [(1, 256), (1, 256), (2, 1)][:])
    assert_true(source.source_bytes().length() > 0)
  }
}

///|
test "export routing execution fixtures" {
  if @env.get_env_var("LUNA_EXPORT_MOE_ROUTING") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN moe-routing")
  for
    (format, score, symbol) in [
      (@format.Format::Bf16, @precision.MoeRoutingScore::Sigmoid, "route_bf16"),
      (F32, SqrtSoftplus, "route_f32"),
    ] {
    println(
      @utf8.decode(
        @moe_routing_cuda_source.routing_source(
          routing_fixture(format, score),
          symbol~,
        ).source_bytes(),
      ),
    )
  }
  println(
    @utf8.decode(
      @moe_routing_cuda_source.routing_source(
        routing_fixture(
          Bf16,
          SqrtSoftplus,
          selection=TokenHash(vocabulary_size=3),
        ),
        symbol="route_hash",
      ).source_bytes(),
    ),
  )
  println("LUNA_SOURCE_END")
}

///|
test "token hash selection uses lookup plus unbiased gathered scores" {
  let source = @moe_routing_cuda_source.routing_source(
    routing_fixture(Bf16, SqrtSoftplus, selection=TokenHash(vocabulary_size=3)),
    symbol="route_hash",
  )
  assert_eq(
    source.symbols()[:],
    [
      "route_hash_project", "route_hash_score", "route_hash_lookup", "route_hash_gather",
    ][:],
  )
  assert_eq(source.launches()[:], [(1, 256), (1, 256), (2, 256), (2, 1)][:])
  assert_eq(source.plan().workspace_bytes(), 96L)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("scores[score_base+indices[output+slot]]"))
  assert_false(text.contains("choice[cell]"))
}
