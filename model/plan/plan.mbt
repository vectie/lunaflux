///|
/// Validated immutable semantic graph consumed by later execution planning.
pub struct ModelPlan {
  identity : @spec.ModelIdentity
  shape_constraints : ShapeConstraints
  workspace_bounds : WorkspaceBounds
  operations : ReadOnlyArray[PlanOperation]
  required_capabilities : ReadOnlyArray[KernelCapabilityId]
  final_output : ActivationRef
} derive(Debug, Eq)

///|
fn expected_capability(kind : OperationKind) -> KernelCapabilityId {
  match kind {
    Embedding => KernelCapabilityId::embedding_lookup()
    RmsNorm => KernelCapabilityId::rms_norm()
    QkvProjection => KernelCapabilityId::qkv_projection()
    RotaryEmbedding => KernelCapabilityId::rotary_embedding()
    CausalAttention => KernelCapabilityId::causal_attention()
    OutputProjection | LanguageModelHead =>
      KernelCapabilityId::dense_projection()
    ResidualAdd => KernelCapabilityId::residual_add()
    GatedMlp => KernelCapabilityId::gated_mlp()
  }
}

///|
fn expected_arity(kind : OperationKind) -> (Int, Int) {
  match kind {
    Embedding => (2, 1)
    RmsNorm => (2, 1)
    QkvProjection => (4, 1)
    RotaryEmbedding => (1, 1)
    CausalAttention => (1, 1)
    OutputProjection => (2, 1)
    ResidualAdd => (2, 1)
    GatedMlp => (4, 1)
    LanguageModelHead => (2, 1)
  }
}

///|
fn valid_input_roles(
  kind : OperationKind,
  inputs : ReadOnlyArray[ValueRef],
) -> Bool {
  match kind {
    Embedding => inputs[0] is ModelInput(TokenIds) && inputs[1] is Tensor(_)
    RmsNorm | OutputProjection | LanguageModelHead =>
      inputs[0] is Activation(_) && inputs[1] is Tensor(_)
    QkvProjection | GatedMlp =>
      inputs[0] is Activation(_) &&
      inputs[1] is Tensor(_) &&
      inputs[2] is Tensor(_) &&
      inputs[3] is Tensor(_)
    RotaryEmbedding | CausalAttention => inputs[0] is Activation(_)
    ResidualAdd => inputs[0] is Activation(_) && inputs[1] is Activation(_)
  }
}

///|
fn valid_operation_shape(
  kind : OperationKind,
  shape : OperationShape,
  constraints : ShapeConstraints,
) -> Bool {
  let hidden = constraints.hidden_size()
  let intermediate = constraints.intermediate_size()
  let vocabulary = constraints.vocabulary_size()
  let query_heads = constraints.attention_head_count()
  let kv_heads = constraints.kv_head_count()
  let head_dimension = constraints.head_dimension()
  let kv_width = kv_heads * head_dimension
  let qkv_width = hidden + 2 * kv_width
  match (kind, shape) {
    (Embedding, EmbeddingShape(vocabulary_size~, width~)) =>
      vocabulary_size == vocabulary && width == hidden
    (RmsNorm, NormalizationShape(width~, epsilon~)) =>
      width == hidden &&
      !epsilon.is_nan() &&
      !epsilon.is_inf() &&
      epsilon > 0.0 &&
      epsilon <= 1.0
    (QkvProjection, ProjectionShape(input_width~, output_width~)) =>
      input_width == hidden && output_width == qkv_width
    (
      RotaryEmbedding,
      RotaryShape(
        query_heads=shape_query_heads,
        kv_heads=shape_kv_heads,
        head_dimension=shape_head_dimension,
        rope_theta~
      ),
    ) =>
      shape_query_heads == query_heads &&
      shape_kv_heads == kv_heads &&
      shape_head_dimension == head_dimension &&
      !rope_theta.is_nan() &&
      rope_theta > 0.0 &&
      rope_theta <= 1.0e12
    (
      CausalAttention,
      AttentionShape(
        query_heads=shape_query_heads,
        kv_heads=shape_kv_heads,
        head_dimension=shape_head_dimension
      ),
    ) =>
      shape_query_heads == query_heads &&
      shape_kv_heads == kv_heads &&
      shape_head_dimension == head_dimension
    (OutputProjection, ProjectionShape(input_width~, output_width~)) =>
      input_width == hidden && output_width == hidden
    (ResidualAdd, ElementwiseShape(width~)) => width == hidden
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~)) =>
      hidden_size == hidden && intermediate_size == intermediate
    (LanguageModelHead, ProjectionShape(input_width~, output_width~)) =>
      input_width == hidden && output_width == vocabulary
    _ => false
  }
}

///|
fn validate_operation(
  operation : PlanOperation,
  index : Int,
  constraints : ShapeConstraints,
  workspace : WorkspaceBounds,
) -> Unit raise PlanValidationError {
  if operation.id().as_int() != index {
    raise InvalidGraph(OperationOrder)
  }
  let (input_count, output_count) = expected_arity(operation.kind())
  if operation.inputs().length() != input_count ||
    operation.outputs().length() != output_count {
    raise InvalidGraph(OperationArity)
  }
  if !valid_input_roles(operation.kind(), operation.inputs()) {
    raise InvalidGraph(OperationInputRoles)
  }
  if operation.required_capability() != expected_capability(operation.kind()) {
    raise InvalidGraph(OperationCapabilityMismatch)
  }
  if !valid_operation_shape(operation.kind(), operation.shape(), constraints) {
    raise InvalidShape(OperationShapeMismatch)
  }
  if operation.workspace_bytes() > workspace.max_temporary_bytes() {
    raise OperationWorkspaceExceedsPlan
  }
}

///|
/// Validates and copies an ordered operation graph into immutable storage.
pub fn ModelPlan::new(
  identity~ : @spec.ModelIdentity,
  shape_constraints~ : ShapeConstraints,
  workspace_bounds~ : WorkspaceBounds,
  operations~ : ArrayView[PlanOperation],
  final_output~ : ActivationRef,
) -> ModelPlan raise PlanValidationError {
  if operations.is_empty() {
    raise InvalidGraph(EmptyOperationList)
  }
  let produced : Array[ActivationRef] = []
  let required : Array[KernelCapabilityId] = []
  for index, operation in operations {
    validate_operation(operation, index, shape_constraints, workspace_bounds)
    for input in operation.inputs() {
      match input {
        Activation(reference) =>
          if !produced.contains(reference) {
            raise InvalidGraph(MissingActivationInput)
          }
        ModelInput(_) | Tensor(_) => ()
      }
    }
    for output in operation.outputs() {
      if output.as_int() != produced.length() {
        raise InvalidGraph(ActivationOutputOrder)
      }
      produced.push(output)
    }
    let capability = operation.required_capability()
    if !required.contains(capability) {
      required.push(capability)
    }
  }
  guard produced.last() is Some(last_output) && last_output == final_output else {
    raise InvalidGraph(FinalOutputNotLast)
  }
  {
    identity,
    shape_constraints,
    workspace_bounds,
    operations: ReadOnlyArray::from_array(operations),
    required_capabilities: ReadOnlyArray::from_array(required),
    final_output,
  }
}

///|
pub fn ModelPlan::identity(self : ModelPlan) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn ModelPlan::shape_constraints(self : ModelPlan) -> ShapeConstraints {
  self.shape_constraints
}

///|
pub fn ModelPlan::workspace_bounds(self : ModelPlan) -> WorkspaceBounds {
  self.workspace_bounds
}

///|
pub fn ModelPlan::operations(self : ModelPlan) -> ReadOnlyArray[PlanOperation] {
  self.operations
}

///|
pub fn ModelPlan::required_capabilities(
  self : ModelPlan,
) -> ReadOnlyArray[KernelCapabilityId] {
  self.required_capabilities
}

///|
pub fn ModelPlan::final_output(self : ModelPlan) -> ActivationRef {
  self.final_output
}
