///|
fn append_line(output : StringBuilder, text : String) -> Unit {
  output.write_string(text)
  output.write_string("\n")
}

///|
fn operation_kind_name(kind : @plan.OperationKind) -> String {
  match kind {
    Embedding => "embedding"
    RmsNorm => "rms_norm"
    QkvProjection => "qkv_projection"
    RotaryEmbedding => "rotary_embedding"
    CausalAttention => "causal_attention"
    OutputProjection => "output_projection"
    ResidualAdd => "residual_add"
    GatedMlp => "gated_mlp"
    LanguageModelHead => "language_model_head"
  }
}

///|
fn append_shape(output : StringBuilder, shape : @plan.OperationShape) -> Unit {
  match shape {
    EmbeddingShape(vocabulary_size~, width~) => {
      append_line(output, "shape: embedding")
      append_line(output, "shape.vocabulary_size: \{vocabulary_size}")
      append_line(output, "shape.width: \{width}")
    }
    NormalizationShape(width~, epsilon~) => {
      append_line(output, "shape: normalization")
      append_line(output, "shape.width: \{width}")
      append_line(
        output,
        "shape.epsilon_bits: \{epsilon.reinterpret_as_uint64()}",
      )
    }
    ProjectionShape(input_width~, output_width~) => {
      append_line(output, "shape: projection")
      append_line(output, "shape.input_width: \{input_width}")
      append_line(output, "shape.output_width: \{output_width}")
    }
    RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~) => {
      append_line(output, "shape: rotary")
      append_line(output, "shape.query_heads: \{query_heads}")
      append_line(output, "shape.kv_heads: \{kv_heads}")
      append_line(output, "shape.head_dimension: \{head_dimension}")
      append_line(
        output,
        "shape.rope_theta_bits: \{rope_theta.reinterpret_as_uint64()}",
      )
    }
    AttentionShape(query_heads~, kv_heads~, head_dimension~) => {
      append_line(output, "shape: attention")
      append_line(output, "shape.query_heads: \{query_heads}")
      append_line(output, "shape.kv_heads: \{kv_heads}")
      append_line(output, "shape.head_dimension: \{head_dimension}")
    }
    ElementwiseShape(width~) => {
      append_line(output, "shape: elementwise")
      append_line(output, "shape.width: \{width}")
    }
    GatedMlpShape(hidden_size~, intermediate_size~) => {
      append_line(output, "shape: gated_mlp")
      append_line(output, "shape.hidden_size: \{hidden_size}")
      append_line(output, "shape.intermediate_size: \{intermediate_size}")
    }
  }
}

///|
fn phase_name(phase : @device_profile.ExecutionPhase) -> String {
  match phase {
    FullPrefill => "full_prefill"
    FullRecompute => "full_recompute"
  }
}

///|
/// Renders immutable, already-admitted AOT metadata. This function performs no
/// signature verification, file access, compilation, JIT, device discovery,
/// module loading, or runtime capability selection.
pub fn render(
  admission : @luna_artifact_admission.LunaAotKernelAdmission,
) -> String {
  let target = admission.target()
  let identity = admission.identity()
  let entry = admission.entry_point()
  let graph = admission.graph_metadata()
  let graph_shape = graph.shape_class()
  let output = StringBuilder()
  append_line(output, "LunaFlux admitted AOT kernel")
  append_line(
    output,
    "manifest sha256: \{admission.manifest_digest().as_hex()}",
  )
  append_line(
    output,
    "specialization sha256: \{admission.specialization_digest().as_hex()}",
  )
  append_line(output, "model content sha256: \{identity.content().as_hex()}")
  append_line(output, "model plan sha256: \{identity.plan().as_hex()}")
  append_line(
    output,
    "target: sm_\{target.compute_major()}\{target.compute_minor()}",
  )
  append_line(output, "target.bf16: \{target.supports_bf16()}")
  append_line(output, "target.cublas_lt: \{target.supports_cublas_lt()}")
  append_line(
    output,
    "catalog version: \{admission.catalog_version().as_int()}",
  )
  append_line(output, "operation id: \{admission.operation_id().as_int()}")
  append_line(
    output,
    "operation kind: \{operation_kind_name(admission.operation_kind())}",
  )
  append_shape(output, admission.operation_shape())
  append_line(output, "capability id: \{admission.capability().as_int()}")
  append_line(output, "entry family id: \{entry.family().family_id().as_int()}")
  append_line(output, "entry point id: \{entry.entry_point_id().as_int()}")
  append_line(output, "module sha256: \{admission.module_digest().sha256()}")
  append_line(
    output,
    "artifact module index: \{admission.artifact_module_index()}",
  )
  append_line(output, "workspace bytes: \{admission.workspace_bytes()}")
  append_line(output, "workspace alignment: \{admission.workspace_alignment()}")
  append_line(
    output,
    "minimum alignment: \{admission.minimum_alignment_bytes()}",
  )
  append_line(
    output,
    "approved source sha256: \{admission.approved_source_identity().as_hex()}",
  )
  append_line(
    output,
    "approved key identity sha256: \{admission.approved_key_identity().as_hex()}",
  )
  append_line(
    output,
    "detached signature sha256: \{admission.detached_signature_digest().as_hex()}",
  )
  append_line(output, "graph.capture_safe: \{graph.capture_safe()}")
  if graph.capture_safe() {
    append_line(output, "graph.memory_accounting: declared_upper_bound")
    append_line(
      output,
      "graph.memory_declared_upper_bound_bytes: \{graph.graph_memory_bytes()}",
    )
    append_line(
      output,
      "graph.memory_alignment_bytes: \{graph.graph_memory_alignment()}",
    )
  } else {
    append_line(output, "graph.memory_accounting: unsupported")
    append_line(
      output, "graph.memory_declared_upper_bound_bytes: unavailable (capture unsupported)",
    )
    append_line(
      output, "graph.memory_alignment_bytes: unavailable (capture unsupported)",
    )
  }
  append_line(
    output,
    "graph.eager_fallback_sha256: \{graph.eager_fallback_digest().as_hex()}",
  )
  append_line(output, "graph.phase: \{phase_name(graph_shape.phase())}")
  append_line(output, "graph.batch_rows: \{graph_shape.batch_rows()}")
  append_line(
    output,
    "graph.query_tokens_per_row: \{graph_shape.query_tokens_per_row()}",
  )
  append_line(output, "graph.context_tokens: \{graph_shape.context_tokens()}")
  append_line(output, "graph.token_rows: \{graph_shape.token_rows()}")
  append_line(output, "execution disposition: validated_eager_fallback")
  output.to_string()
}
