///|
async test "original routing replay exports full literal hash and learned rows" {
  let representation = @format.Representation::new(
    format=Bf16,
    rows=256,
    columns=4096,
  )
  let weight = @precision.plan_operand(
    role=Weight,
    representation~,
    placement=NativeOperand,
    supported_placements=[NativeOperand],
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    memory_ceiling=representation.payload_bytes(),
  )
  let output = StringBuilder()
  for
    (symbol, selection) in [
      (
        "matched_hash",
        @precision.MoeRoutingSelection::TokenHash(vocabulary_size=129280),
      ),
      ("matched_learned", @precision.MoeRoutingSelection::GroupedTopK),
    ] {
    let plan = @precision.MoeRoutingPrecision::new(
      weight~,
      maximum_rows=144,
      group_count=1,
      group_score_topk=1,
      selected_group_count=1,
      selected_experts=6,
      score=SqrtSoftplus,
      selection~,
      normalize=true,
      scaling=1.5,
      maximum_workspace_bytes=1048576L,
    )
    let source = @moe_routing_cuda_source.routing_source(plan, symbol~)
    assert_eq(source.launches()[0], (144, 256))
    assert_eq(source.launches()[1], (144, 256))
    assert_eq(source.launches().last().unwrap(), (144, 1))
    output.write_string(@utf8.decode(source.source_bytes()))
    output.write_string("\n")
  }
  if @env.get_env_var("LUNAFLUX_MATCHED_ROUTER_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(
      path,
      @utf8.encode(output.to_string()),
      create_mode=CreateNew,
    )
  }
}
