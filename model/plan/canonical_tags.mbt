///|
fn execution_mode_tag(mode : ModelKvExecution) -> Byte {
  match mode {
    StatelessFullContext => 0x01
    PagedKeyValue => 0x02
  }
}

///|
fn model_dtype_tag(dtype : @spec.ModelDType) -> Byte {
  match dtype {
    BFloat16 => 0x01
    Float16 => 0x02
    Float32 => 0x03
  }
}

///|
fn kv_cache_layout_tag(layout : KvCacheLayout) -> Byte {
  match layout {
    LayerMajorSplitKeyValueV1 => 0x01
  }
}

///|
fn value_ref_tag(value : ValueRef) -> Byte {
  match value {
    ModelInput(_) => 0x01
    Tensor(_) => 0x02
    Activation(_) => 0x03
  }
}

///|
fn operation_kind_tag(kind : OperationKind) -> Byte {
  match kind {
    Embedding => 0x01
    RmsNorm => 0x02
    QkvProjection => 0x03
    QkRmsNorm => 0x0a
    RotaryEmbedding => 0x04
    CausalAttention => 0x05
    OutputProjection => 0x06
    ResidualAdd => 0x07
    GatedMlp => 0x08
    LanguageModelHead => 0x09
  }
}

///|
fn operation_shape_tag(shape : OperationShape) -> Byte {
  match shape {
    EmbeddingShape(..) => 0x01
    NormalizationShape(..) => 0x02
    QkNormalizationShape(..) => 0x08
    ProjectionShape(..) => 0x03
    RotaryShape(..) => 0x04
    AttentionShape(..) => 0x05
    ElementwiseShape(..) => 0x06
    GatedMlpShape(..) => 0x07
  }
}

///|
fn runtime_input_tag(input : OperationRuntimeInput) -> Byte {
  match input {
    StepCounts => 0x01
    QueryPositions => 0x02
    QueryRowOffsets => 0x03
    SequenceLengths => 0x04
    PageTableOffsets => 0x05
    PageTablePageIndices => 0x06
  }
}

///|
fn optional_kv_layer_tag(layer : KvLayerId?) -> Byte {
  match layer {
    None => 0x00
    Some(_) => 0x01
  }
}

///|
fn persistent_state_tag(state : PersistentStateAccess) -> Byte {
  match state {
    NoPersistentState => 0x00
    KvCacheReadWrite(_) => 0x01
  }
}
