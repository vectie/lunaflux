///|
fn tile_value(id : Int) -> AttentionTileValueId {
  AttentionTileValueId(id)
}

///|
fn tile_binding(
  id : Int,
  kind : AttentionTileValueKind,
  expression : AttentionTileExpression,
) -> AttentionTileBinding {
  { id: tile_value(id), kind, expression }
}

///|
fn expression_dependencies(
  expression : AttentionTileExpression,
) -> Array[AttentionTileValueId] {
  match expression {
    AttentionTileInput(_) => []
    AttentionTileDot(left, right) => [left, right]
    AttentionTileScale(input)
    | AttentionTileCausalMask(input)
    | AttentionTileFoldKeyValueStream(input)
    | AttentionTileNormalize(input) => [input]
    AttentionTileOnlineSoftmax(scores~, values~) => [scores, values]
  }
}

///|
fn binding_kind_before(
  bindings : ReadOnlyArray[AttentionTileBinding],
  index : Int,
  id : AttentionTileValueId,
) -> AttentionTileValueKind? {
  for candidate_index in 0..<index {
    let candidate = bindings[candidate_index]
    if candidate.id == id {
      return Some(candidate.kind)
    }
  }
  None
}

///|
fn expected_expression_kind(
  bindings : ReadOnlyArray[AttentionTileBinding],
  index : Int,
  expression : AttentionTileExpression,
) -> AttentionTileValueKind? {
  match expression {
    AttentionTileInput(QueryTileStream) => Some(QueryTileValue)
    AttentionTileInput(PagedKeyTileStream) => Some(KeyTileValue)
    AttentionTileInput(PagedValueTileStream) => Some(ValueTileValue)
    AttentionTileDot(left, right) =>
      if binding_kind_before(bindings, index, left) == Some(QueryTileValue) &&
        binding_kind_before(bindings, index, right) == Some(KeyTileValue) {
        Some(ScoreTileValue)
      } else {
        None
      }
    AttentionTileScale(input) | AttentionTileCausalMask(input) =>
      if binding_kind_before(bindings, index, input) == Some(ScoreTileValue) {
        Some(ScoreTileValue)
      } else {
        None
      }
    AttentionTileOnlineSoftmax(scores~, values~) =>
      if binding_kind_before(bindings, index, scores) == Some(ScoreTileValue) &&
        binding_kind_before(bindings, index, values) == Some(ValueTileValue) {
        Some(OnlineSoftmaxStateValue)
      } else {
        None
      }
    AttentionTileFoldKeyValueStream(input) =>
      if binding_kind_before(bindings, index, input) ==
        Some(OnlineSoftmaxStateValue) {
        Some(OnlineSoftmaxStateValue)
      } else {
        None
      }
    AttentionTileNormalize(input) =>
      if binding_kind_before(bindings, index, input) ==
        Some(OnlineSoftmaxStateValue) {
        Some(OutputTileValue)
      } else {
        None
      }
  }
}

///|
fn validate_attention_tile_bindings(
  bindings : ReadOnlyArray[AttentionTileBinding],
  root : AttentionTileValueId,
) -> Unit raise AttentionTileIrError {
  if bindings.length() == 0 || bindings[bindings.length() - 1].id != root {
    raise InvalidAttentionTileIr(InvalidRoot)
  }
  for index, binding in bindings {
    if binding.id.as_int() != index + 1 {
      raise InvalidAttentionTileIr(InvalidTopology)
    }
    for dependency in expression_dependencies(binding.expression) {
      if dependency.as_int() >= binding.id.as_int() {
        raise InvalidAttentionTileIr(InvalidTopology)
      }
    }
    if expected_expression_kind(bindings, index, binding.expression) !=
      Some(binding.kind) {
      raise InvalidAttentionTileIr(InvalidValueKind)
    }
  }
  if bindings[bindings.length() - 1].kind != OutputTileValue {
    raise InvalidAttentionTileIr(InvalidRoot)
  }
}

///|
fn attention_phase_code(
  phase : @luna_attention_strategy.AttentionExecutionPhase,
) -> Int {
  match phase {
    Prefill => 1
    Decode => 2
    MixedCompatibility => 3
  }
}

///|
fn attention_numeric_code(
  numeric_type : @luna_attention_strategy.AttentionNumericType,
) -> Int {
  match numeric_type {
    AttentionBFloat16 => 1
    AttentionFloat16 => 2
    AttentionFloat32 => 3
    AttentionFloat8 => 4
  }
}

///|
fn attention_compute_code(
  compute : @luna_attention_strategy.AttentionTileCompute,
) -> Int {
  match compute {
    AttentionScalarDot => 1
    AttentionSubgroupDot => 2
    AttentionMatrixDot => 3
    AttentionMatrixQkPv => 4
  }
}

///|
fn attention_memory_code(
  memory : @luna_attention_strategy.AttentionTileMemory,
) -> Int {
  match memory {
    AttentionDirectGlobal => 1
    AttentionStagedShared => 2
    AttentionAsyncStagedShared => 3
  }
}

///|
fn attention_expression_code(expression : AttentionTileExpression) -> String {
  match expression {
    AttentionTileInput(role) =>
      "input:" +
      (match role {
        QueryTileStream => "q"
        PagedKeyTileStream => "k"
        PagedValueTileStream => "v"
      })
    AttentionTileDot(left, right) => "dot:\{left.as_int()}:\{right.as_int()}"
    AttentionTileScale(input) => "scale:\{input.as_int()}"
    AttentionTileCausalMask(input) => "causal:\{input.as_int()}"
    AttentionTileOnlineSoftmax(scores~, values~) =>
      "online:\{scores.as_int()}:\{values.as_int()}"
    AttentionTileFoldKeyValueStream(input) => "fold-kv:\{input.as_int()}"
    AttentionTileNormalize(input) => "normalize:\{input.as_int()}"
  }
}

///|
fn canonical_attention_tile_program(
  problem : @luna_attention_strategy.AttentionCompileProblem,
  candidate : @luna_attention_strategy.AttentionTileCandidate,
  bindings : ReadOnlyArray[AttentionTileBinding],
  root : AttentionTileValueId,
) -> Bytes {
  let (query_heads, key_value_heads, head_dimension, tokens_per_page) = problem
    .shape()
    .dimensions()
  let (query_rows, context_tokens) = problem.dimensions()
  let (tile_query, tile_key_value, tile_head) = candidate.tile().dimensions()
  let (subgroups, stages, vector_bytes) = candidate.execution_geometry()
  let builder = StringBuilder()
  builder <+ "luna-attention-tile-ir-v1\n"
  builder <+
    "shape=\{query_heads},\{key_value_heads},\{head_dimension},\{tokens_per_page}\n"
  builder <+
    "problem=\{attention_phase_code(problem.phase())},\{attention_numeric_code(problem.numeric_type())},\{query_rows},\{context_tokens},\{problem.is_causal()}\n"
  builder <+
    "candidate=\{candidate.stable_id()},\{attention_compute_code(candidate.compute())},\{attention_memory_code(candidate.memory())}\n"
  builder <+
    "tile=\{tile_query},\{tile_key_value},\{tile_head},\{subgroups},\{stages},\{vector_bytes},\{candidate.shared_memory_bytes()}\n"
  for binding in bindings {
    builder <+
      "bind=\{binding.id.as_int()}:\{attention_expression_code(binding.expression)}\n"
  }
  builder <+ "root=\{root.as_int()}\n"
  @utf8.encode(builder.to_string())
}

///|
/// Lowers a selected strategy into a pure functional attention tile program.
/// Device scheduling and instruction selection consume this immutable dialect;
/// they are not encoded in the model graph or attention semantics.
pub fn compile_attention_tile_ir(
  plan : @luna_attention_strategy.CompiledAttentionTilePlan,
) -> AttentionTileProgram raise AttentionTileIrError {
  let problem = plan.problem()
  let candidate = plan.candidate()
  let bindings : Array[AttentionTileBinding] = if problem.is_causal() {
    [
      tile_binding(1, QueryTileValue, AttentionTileInput(QueryTileStream)),
      tile_binding(2, KeyTileValue, AttentionTileInput(PagedKeyTileStream)),
      tile_binding(3, ValueTileValue, AttentionTileInput(PagedValueTileStream)),
      tile_binding(
        4,
        ScoreTileValue,
        AttentionTileDot(tile_value(1), tile_value(2)),
      ),
      tile_binding(5, ScoreTileValue, AttentionTileScale(tile_value(4))),
      tile_binding(6, ScoreTileValue, AttentionTileCausalMask(tile_value(5))),
      tile_binding(
        7,
        OnlineSoftmaxStateValue,
        AttentionTileOnlineSoftmax(scores=tile_value(6), values=tile_value(3)),
      ),
      tile_binding(
        8,
        OnlineSoftmaxStateValue,
        AttentionTileFoldKeyValueStream(tile_value(7)),
      ),
      tile_binding(9, OutputTileValue, AttentionTileNormalize(tile_value(8))),
    ]
  } else {
    [
      tile_binding(1, QueryTileValue, AttentionTileInput(QueryTileStream)),
      tile_binding(2, KeyTileValue, AttentionTileInput(PagedKeyTileStream)),
      tile_binding(3, ValueTileValue, AttentionTileInput(PagedValueTileStream)),
      tile_binding(
        4,
        ScoreTileValue,
        AttentionTileDot(tile_value(1), tile_value(2)),
      ),
      tile_binding(5, ScoreTileValue, AttentionTileScale(tile_value(4))),
      tile_binding(
        6,
        OnlineSoftmaxStateValue,
        AttentionTileOnlineSoftmax(scores=tile_value(5), values=tile_value(3)),
      ),
      tile_binding(
        7,
        OnlineSoftmaxStateValue,
        AttentionTileFoldKeyValueStream(tile_value(6)),
      ),
      tile_binding(8, OutputTileValue, AttentionTileNormalize(tile_value(7))),
    ]
  }
  let immutable_bindings = ReadOnlyArray::from_array(bindings)
  let root = immutable_bindings[immutable_bindings.length() - 1].id
  validate_attention_tile_bindings(immutable_bindings, root)
  let canonical = canonical_attention_tile_program(
    problem, candidate, immutable_bindings, root,
  )
  {
    problem,
    candidate,
    bindings: immutable_bindings,
    root,
    canonical,
    digest: AttentionTileProgramDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  }
}
