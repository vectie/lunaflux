///|
fn candidate_family_name(candidate : LunaBf16KernelCandidate) -> String {
  match candidate {
    Pointwise(value) =>
      match value.family() {
        EmbeddingLookup => "embedding_lookup"
        RmsNorm => "rms_norm"
        QkRmsNorm => "qk_rms_norm"
        PositionedRotary => "positioned_rotary"
        ResidualAdd => "residual_add"
        KvCacheWrite => "kv_cache_write"
      }
    Projection(value) =>
      match value.family() {
        QkvProjection => "qkv_projection"
        DenseProjection => "dense_projection"
        GatedMlp => "gated_mlp"
        LanguageModelHead => "language_model_head"
      }
    PagedAttention(_) => "paged_attention"
  }
}

///|
fn candidate_source_bytes(candidate : LunaBf16KernelCandidate) -> Bytes {
  match candidate {
    Pointwise(value) => value.source_bytes()
    Projection(value) => value.source_bytes()
    PagedAttention(value) => value.source_bytes()
  }
}

///|
fn candidate_recipe_bytes(candidate : LunaBf16KernelCandidate) -> Bytes {
  match candidate {
    Pointwise(value) => value.recipe_bytes()
    Projection(value) => value.recipe_bytes()
    PagedAttention(value) => value.recipe_bytes()
  }
}

///|
fn candidate_recipe_digest(candidate : LunaBf16KernelCandidate) -> String {
  match candidate {
    Pointwise(value) => value.recipe_digest().as_hex()
    Projection(value) => value.recipe_digest().as_hex()
    PagedAttention(value) => value.recipe_digest().as_hex()
  }
}

///|
fn candidate_target_name(
  target : @catalog.DeviceTarget,
) -> String raise LunaBf16KernelProducerError {
  if target.compute_major() <= 0 ||
    target.compute_minor() < 0 ||
    target.compute_minor() > 9 {
    raise Invalid(DeviceTarget)
  }
  "sm_" +
  target.compute_major().to_string() +
  target.compute_minor().to_string()
}

///|
fn candidate_key(operation_id : Int, family : String) -> String {
  let id = operation_id.to_string()
  "op_" + "0".repeat(8 - id.length()) + id + "_" + family
}

///|
fn append_candidate_line(output : StringBuilder, value : String) -> Unit {
  output.write_string(value)
  output.write_string("\n")
}

///|
fn validate_candidate_sequence(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  candidates : ArrayView[LunaBf16KernelCandidate],
) -> Unit raise LunaBf16KernelProducerError {
  if candidates.length() != model_plan.operations().length() {
    raise Mismatch(OperationCoverage)
  }
  for index, operation in model_plan.operations() {
    let candidate = candidates[index]
    if operation.id().as_int() != index ||
      candidate_operation_id(candidate) != operation.id() {
      raise Mismatch(OperationIdentity)
    }
    if candidate_target(candidate) != target {
      raise Mismatch(DeviceTarget)
    }
    if !candidate_kind_matches(candidate, operation.kind()) {
      raise Mismatch(OperationKind)
    }
    let source = candidate_source_bytes(candidate)
    let recipe = candidate_recipe_bytes(candidate)
    if sha256_bytes(source) != candidate_source_digest(candidate) {
      raise Mismatch(CandidateSource)
    }
    if sha256_bytes(recipe) != candidate_recipe_digest(candidate) {
      raise Mismatch(CandidateRecipe)
    }
  }
}

///|
fn lower_candidate(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : LunaBf16CandidateSetLimits,
) -> LunaBf16KernelCandidate raise LunaBf16KernelProducerError {
  let operands = derive_candidate_operands(
    model_plan, operation, profile, layout,
  )
  match operation.kind() {
    Embedding | RmsNorm | QkRmsNorm | RotaryEmbedding | ResidualAdd =>
      Pointwise(
        @luna_cuda_pointwise_aot.lower_reusable_pointwise_cuda_aot_candidate(
          operation,
          profile,
          target,
          compiler,
          operands,
          @luna_cuda_pointwise_aot.PointwiseAotLimits::new(
            max_source_bytes=limits.max_source_bytes,
            max_recipe_bytes=limits.max_recipe_bytes,
          ),
        ) catch {
          _ => raise Invalid(CandidateSet)
        },
      )
    QkvProjection | OutputProjection | GatedMlp | LanguageModelHead =>
      Projection(
        @luna_cuda_projection_aot.lower_reusable_projection_cuda_aot_candidate(
          operation,
          profile,
          target,
          compiler,
          operands,
          @luna_cuda_projection_aot.ProjectionAotLimits::new(
            max_source_bytes=limits.max_source_bytes,
            max_recipe_bytes=limits.max_recipe_bytes,
          ),
        ) catch {
          _ => raise Invalid(CandidateSet)
        },
      )
    CausalAttention =>
      PagedAttention(
        @luna_cuda_paged_attention_aot.lower_reusable_paged_attention_cuda_aot_candidate(
          operation,
          profile,
          layout,
          target,
          compiler,
          operands,
          @luna_cuda_paged_attention_aot.PagedAttentionAotLimits::new(
            max_source_bytes=limits.max_source_bytes,
            max_recipe_bytes=limits.max_recipe_bytes,
          ),
        ) catch {
          _ => raise Invalid(CandidateSet)
        },
      )
  }
}

///|
fn build_candidate_declaration(
  target : @catalog.DeviceTarget,
  payloads : ArrayView[LunaBf16CandidatePayload],
) -> Bytes raise LunaBf16KernelProducerError {
  let output = StringBuilder()
  append_candidate_line(output, "schema=lunaflux-bf16-candidate-set.v1")
  append_candidate_line(output, "target=" + candidate_target_name(target))
  append_candidate_line(
    output,
    "operation_count=" + payloads.length().to_string(),
  )
  for index, payload in payloads {
    append_candidate_line(
      output,
      "candidate=" +
      payload.key +
      "," +
      index.to_string() +
      "," +
      payload.family,
    )
  }
  @utf8.encode(output.to_string())
}

///|
fn build_candidate_inventory(
  declaration_sha256 : String,
  payloads : ArrayView[LunaBf16CandidatePayload],
) -> Bytes {
  let output = StringBuilder()
  append_candidate_line(output, declaration_sha256 + "  candidate-set.v1")
  for payload in payloads {
    append_candidate_line(
      output,
      payload.recipe_sha256 + "  recipes/" + payload.key + ".recipe",
    )
  }
  for payload in payloads {
    append_candidate_line(
      output,
      payload.source_sha256 + "  sources/" + payload.key + ".cu",
    )
  }
  @utf8.encode(output.to_string())
}

///|
/// Derives the complete remote-compiler candidate set from one authenticated
/// model plan. Source, symbols, operands, recipes, ordering, and file keys are
/// all product-generated; callers provide no textual kernel input.
pub fn prepare_luna_bf16_candidate_set(
  model_plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : LunaBf16CandidateSetLimits,
) -> LunaBf16CandidateSet raise LunaBf16KernelProducerError {
  let operations = model_plan.operations()
  if operations.length() < 9 || operations.length() > limits.max_operations {
    raise LimitExceeded(OperationCoverage, maximum=limits.max_operations)
  }
  if !layout.is_canonical_for(
      model_plan.identity(),
      model_plan.kv_cache_geometry(),
    ) ||
    !target.supports_bf16() ||
    compiler.reassociate() {
    raise Mismatch(CandidateSet)
  }
  let shape = model_plan.shape_constraints()
  if profile.max_query_rows() != shape.max_batch_rows() ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_query_tokens() > shape.max_sequence_tokens() ||
    profile.max_page_table_entries() > layout.total_page_count() {
    raise Mismatch(Profile)
  }
  require_complete_kinds(operations)
  let candidates : Array[LunaBf16KernelCandidate] = []
  for index, operation in operations {
    if operation.id().as_int() != index {
      raise Mismatch(OperationIdentity)
    }
    candidates.push(
      lower_candidate(
        model_plan, operation, layout, profile, target, compiler, limits,
      ),
    )
  }
  validate_candidate_sequence(model_plan, target, candidates)
  let payloads : Array[LunaBf16CandidatePayload] = []
  let mut total_bytes = 0
  for candidate in candidates {
    let source = candidate_source_bytes(candidate)
    let recipe = candidate_recipe_bytes(candidate)
    if source.length() > limits.max_source_bytes {
      raise LimitExceeded(CandidateSource, maximum=limits.max_source_bytes)
    }
    if recipe.length() > limits.max_recipe_bytes {
      raise LimitExceeded(CandidateRecipe, maximum=limits.max_recipe_bytes)
    }
    if total_bytes > limits.max_total_bytes - source.length() ||
      total_bytes + source.length() > limits.max_total_bytes - recipe.length() {
      raise LimitExceeded(CandidateSet, maximum=limits.max_total_bytes)
    }
    total_bytes += source.length() + recipe.length()
    let family = candidate_family_name(candidate)
    payloads.push({
      key: candidate_key(candidate_operation_id(candidate).as_int(), family),
      family,
      source,
      source_sha256: candidate_source_digest(candidate),
      recipe,
      recipe_sha256: candidate_recipe_digest(candidate),
    })
  }
  let declaration = build_candidate_declaration(target, payloads)
  let declaration_sha256 = sha256_bytes(declaration)
  let inventory = build_candidate_inventory(declaration_sha256, payloads)
  if total_bytes > limits.max_total_bytes - declaration.length() ||
    total_bytes + declaration.length() >
    limits.max_total_bytes - inventory.length() {
    raise LimitExceeded(CandidateInventory, maximum=limits.max_total_bytes)
  }
  total_bytes += declaration.length() + inventory.length()
  {
    identity: model_plan.identity(),
    target,
    profile,
    compiler,
    candidates: ReadOnlyArray::from_array(candidates),
    payloads: ReadOnlyArray::from_array(payloads),
    declaration,
    declaration_sha256,
    inventory,
    inventory_sha256: sha256_bytes(inventory),
    total_bytes,
  }
}
