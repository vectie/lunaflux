///|
fn add_tensor(
  tensors : Array[Glm53TensorDescriptor],
  name : String,
  dtype : Glm53TensorDType,
  shape : ArrayView[Int64],
) -> Unit {
  tensors.push({
    ordinal: tensors.length(),
    name,
    dtype,
    shape: ReadOnlyArray::from_array(shape),
  })
}

///|
fn add_dense_mlp(
  tensors : Array[Glm53TensorDescriptor],
  prefix : String,
  hidden : Int64,
  intermediate : Int64,
) -> Unit {
  add_tensor(tensors, "\{prefix}.gate_proj.weight", BFloat16, [
    intermediate, hidden,
  ])
  add_tensor(tensors, "\{prefix}.up_proj.weight", BFloat16, [
    intermediate, hidden,
  ])
  add_tensor(tensors, "\{prefix}.down_proj.weight", BFloat16, [
    hidden, intermediate,
  ])
}

///|
fn add_moe(
  tensors : Array[Glm53TensorDescriptor],
  prefix : String,
  hidden : Int64,
  intermediate : Int64,
  expert_count : Int,
) -> Unit {
  add_tensor(tensors, "\{prefix}.gate.weight", BFloat16, [
    Int64::from_int(expert_count),
    hidden,
  ])
  add_tensor(tensors, "\{prefix}.gate.e_score_correction_bias", Float32, [
    Int64::from_int(expert_count),
  ])
  for expert in 0..<expert_count {
    let expert_prefix = "\{prefix}.experts.\{expert}"
    add_dense_mlp(tensors, expert_prefix, hidden, intermediate)
  }
  add_dense_mlp(tensors, "\{prefix}.shared_experts", hidden, intermediate)
}

///|
fn add_dsa(
  tensors : Array[Glm53TensorDescriptor],
  prefix : String,
  hidden : Int64,
  heads : Int64,
  q_rank : Int64,
  kv_rank : Int64,
  nope : Int64,
  rope : Int64,
  value : Int64,
  add_index : Bool,
  add_pool : Bool,
) -> Unit {
  add_tensor(tensors, "\{prefix}.q_a_proj.weight", BFloat16, [q_rank, hidden])
  add_tensor(tensors, "\{prefix}.q_a_layernorm.weight", BFloat16, [q_rank])
  add_tensor(tensors, "\{prefix}.q_b_proj.weight", BFloat16, [
    heads * (nope + rope),
    q_rank,
  ])
  add_tensor(tensors, "\{prefix}.kv_a_proj_with_mqa.weight", BFloat16, [
    kv_rank + rope,
    hidden,
  ])
  add_tensor(tensors, "\{prefix}.kv_a_layernorm.weight", BFloat16, [kv_rank])
  add_tensor(tensors, "\{prefix}.kv_b_proj.weight", BFloat16, [
    heads * (nope + value),
    kv_rank,
  ])
  add_tensor(tensors, "\{prefix}.o_proj.weight", BFloat16, [
    hidden,
    heads * value,
  ])
  if add_index {
    add_tensor(tensors, "\{prefix}.indexer.wq_b.weight", BFloat16, [
      4096L, q_rank,
    ])
    add_tensor(tensors, "\{prefix}.indexer.wk.weight", BFloat16, [128L, hidden])
    add_tensor(tensors, "\{prefix}.indexer.k_norm.weight", BFloat16, [128L])
    add_tensor(tensors, "\{prefix}.indexer.k_norm.bias", BFloat16, [128L])
    add_tensor(tensors, "\{prefix}.indexer.weights_proj.weight", BFloat16, [
      32L, hidden,
    ])
    if add_pool {
      add_tensor(
        tensors,
        "\{prefix}.indexer.index_kpool_compress_ape",
        BFloat16,
        [4L, 128L],
      )
      add_tensor(
        tensors,
        "\{prefix}.indexer.index_kpool_compress_gate",
        BFloat16,
        [128L, hidden],
      )
    }
  }
}
