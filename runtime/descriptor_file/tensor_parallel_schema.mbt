///|
fn parse_tensor_parallel_ranks(
  root : Map[String, Json],
) -> ReadOnlyArray[TensorParallelRankDescriptorClaims] raise RuntimeDescriptorFileError {
  guard root.get("ranks") is Some(Array(values)) else {
    raise invalid_field(TensorParallel, ExpectedObject, index=0)
  }
  if values.length() < 2 || values.length() > 16 {
    raise invalid_field(TensorParallel, UnsupportedValue, index=0)
  }
  let ranks : Array[TensorParallelRankDescriptorClaims] = []
  for index, value in values {
    guard value is Object(rank) else {
      raise invalid_field(TensorParallelRank, ExpectedObject, index~)
    }
    validate_fields(rank, TensorParallelRank, [
      "rank", "device_ordinal", "memory_ceiling_bytes", "weight_reserved_bytes",
      "activation_workspace_reserved_bytes", "kv_reserved_bytes", "collective_reserved_bytes",
    ])
    let declared_rank = require_int(rank, "rank", TensorParallelRank)
    if declared_rank != index {
      raise invalid_field(TensorParallelRank, UnsupportedValue, index~)
    }
    ranks.push({
      rank: declared_rank,
      device_ordinal: require_int(
        rank,
        "device_ordinal",
        TensorParallelRank,
        index=1,
      ),
      memory_ceiling_bytes: require_int64(
        rank,
        "memory_ceiling_bytes",
        TensorParallelRank,
        index=2,
      ),
      weight_reserved_bytes: require_int64(
        rank,
        "weight_reserved_bytes",
        TensorParallelRank,
        index=3,
      ),
      activation_workspace_reserved_bytes: require_int64(
        rank,
        "activation_workspace_reserved_bytes",
        TensorParallelRank,
        index=4,
      ),
      kv_reserved_bytes: require_int64(
        rank,
        "kv_reserved_bytes",
        TensorParallelRank,
        index=5,
      ),
      collective_reserved_bytes: require_int64(
        rank,
        "collective_reserved_bytes",
        TensorParallelRank,
        index=6,
      ),
    })
  }
  ReadOnlyArray::from_array(ranks)
}

///|
fn parse_tensor_parallel_section(
  root : Map[String, Json],
) -> TensorParallelDescriptorClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "tensor_parallel", Root, index=7)
  validate_fields(object, TensorParallel, [
    "ranks", "minimum_collective_runtime_version_code", "maximum_collective_runtime_version_code",
    "collective_exchange_timeout_millis",
  ])
  {
    ranks: parse_tensor_parallel_ranks(object),
    minimum_collective_runtime_version_code: require_int(
      object,
      "minimum_collective_runtime_version_code",
      TensorParallel,
      index=1,
    ),
    maximum_collective_runtime_version_code: require_int(
      object,
      "maximum_collective_runtime_version_code",
      TensorParallel,
      index=2,
    ),
    collective_exchange_timeout_millis: require_int(
      object,
      "collective_exchange_timeout_millis",
      TensorParallel,
      index=3,
    ),
  }
}

///|
fn parse_tensor_parallel_kernels(
  root : Map[String, Json],
) -> TensorParallelRuntimeDescriptorKernelClaims raise RuntimeDescriptorFileError {
  let object = require_object(root, "kernels", Root, index=2)
  validate_fields(object, Kernels, [
    "manifest_locator", "manifest_sha256", "policy",
  ])
  {
    manifest_locator: require_string(
      object,
      "manifest_locator",
      Kernels,
      index=0,
    ),
    manifest_digest: require_string(object, "manifest_sha256", Kernels, index=1),
    policy: require_string(object, "policy", Kernels, index=2),
  }
}

///|
fn parse_tensor_parallel_descriptor(
  bytes : BytesView,
  limits : RuntimeDescriptorFileLimits,
) -> TensorParallelRuntimeDescriptorClaims raise RuntimeDescriptorFileError {
  if bytes.length().to_int64() > limits.max_descriptor_bytes() {
    raise FileTooLarge(maximum=limits.max_descriptor_bytes())
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidUtf8 }
  @json_guard.reject_duplicate_object_keys(
    text,
    max_depth=limits.max_json_depth(),
  ) catch {
    InvalidDepthLimit => raise MalformedJson
    DepthLimitExceeded(_) =>
      raise JsonDepthExceeded(maximum=limits.max_json_depth())
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let value = @json.parse(text) catch { _ => raise MalformedJson }
  guard value is Object(root) else { raise MalformedJson }
  validate_fields(root, Root, [
    "schema_version", "model", "kernels", "execution", "worker_limits", "inference_limits",
    "ceilings", "tensor_parallel",
  ])
  if require_string(root, "schema_version", Root) !=
    "lunaflux.runtime.tensor-parallel.v1" {
    raise invalid_field(Root, UnsupportedValue)
  }
  {
    model: parse_model(root),
    kernels: parse_tensor_parallel_kernels(root),
    execution: parse_execution(root),
    worker: parse_worker(root),
    inference: parse_inference(root),
    ceilings: parse_ceilings(root),
    tensor_parallel: parse_tensor_parallel_section(root),
  }
}
