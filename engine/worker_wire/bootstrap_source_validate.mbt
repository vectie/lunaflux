///|
const BOOTSTRAP_SOURCE_MAX_PATH_BYTES : Int = 4096

///|
const BOOTSTRAP_SOURCE_MAX_MODEL_CONFIG_BYTES : Int = 16777216

///|
const BOOTSTRAP_SOURCE_MAX_MANIFEST_BYTES : Int64 = 67108864L

///|
const BOOTSTRAP_SOURCE_MAX_DEVICE_BYTES : Int64 = 8796093022208L

///|
const BOOTSTRAP_SOURCE_MAX_MODULE_BYTES : Int64 = 2147483647L

///|
fn bootstrap_source_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn validate_bootstrap_path_segments(
  value : StringView,
  start : Int,
  rule : WorkerBootstrapSourceRule,
) -> Unit raise WorkerWireError {
  let mut segment_length = 0
  let mut segment_dots = 0
  for index in start..<value.length() {
    let unit = value[index]
    if unit == '/' {
      if segment_length == 0 ||
        (segment_length <= 2 && segment_dots == segment_length) {
        raise InvalidBootstrapSource(rule)
      }
      segment_length = 0
      segment_dots = 0
    } else {
      let valid = unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.')
      if !valid {
        raise InvalidBootstrapSource(rule)
      }
      segment_length += 1
      if unit == '.' {
        segment_dots += 1
      }
    }
  }
  if segment_length == 0 ||
    (segment_length <= 2 && segment_dots == segment_length) {
    raise InvalidBootstrapSource(rule)
  }
}

///|
fn validate_bootstrap_root(value : StringView) -> Unit raise WorkerWireError {
  if value.length() < 2 ||
    value.length() > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
    value[0] != '/' {
    raise InvalidBootstrapSource(RootPath)
  }
  validate_bootstrap_path_segments(value, 1, RootPath)
}

///|
fn validate_bootstrap_locator(value : StringView) -> Unit raise WorkerWireError {
  if value.is_empty() ||
    value.length() > BOOTSTRAP_SOURCE_MAX_PATH_BYTES ||
    value[0] == '/' {
    raise InvalidBootstrapSource(RelativeLocator)
  }
  validate_bootstrap_path_segments(value, 0, RelativeLocator)
}

///|
pub fn WorkerBootstrapRoots::new(
  model_root~ : StringView,
  kernel_root~ : StringView,
) -> WorkerBootstrapRoots raise WorkerWireError {
  validate_bootstrap_root(model_root)
  validate_bootstrap_root(kernel_root)
  { model_root: model_root.to_owned(), kernel_root: kernel_root.to_owned() }
}

///|
pub fn WorkerBootstrapModelSource::new(
  config_locator~ : StringView,
  weights_locator~ : StringView,
  config_digest_sha256~ : StringView,
) -> WorkerBootstrapModelSource raise WorkerWireError {
  validate_bootstrap_locator(config_locator)
  validate_bootstrap_locator(weights_locator)
  if config_locator == weights_locator {
    raise InvalidBootstrapSource(DuplicateLocator)
  }
  if !bootstrap_source_is_sha256(config_digest_sha256) {
    raise InvalidBootstrapSource(Digest)
  }
  {
    config_locator: config_locator.to_owned(),
    weights_locator: weights_locator.to_owned(),
    config_digest: config_digest_sha256.to_owned(),
  }
}

///|
pub fn WorkerBootstrapKernelSource::new(
  manifest_locator~ : StringView,
  manifest_digest_sha256~ : StringView,
) -> WorkerBootstrapKernelSource raise WorkerWireError {
  validate_bootstrap_locator(manifest_locator)
  if !bootstrap_source_is_sha256(manifest_digest_sha256) {
    raise InvalidBootstrapSource(Digest)
  }
  {
    manifest_locator: manifest_locator.to_owned(),
    manifest_digest: manifest_digest_sha256.to_owned(),
  }
}

///|
pub fn WorkerBootstrapExecution::new(
  compute_major~ : Int,
  compute_minor~ : Int,
  supports_bf16~ : Bool,
  supports_cublas_lt~ : Bool,
  tokens_per_page~ : Int,
  total_page_count~ : Int,
) -> WorkerBootstrapExecution raise WorkerWireError {
  if compute_major <= 0 ||
    compute_major > 1024 ||
    compute_minor < 0 ||
    compute_minor > 1024 {
    raise InvalidBootstrapSource(Target)
  }
  if tokens_per_page <= 0 ||
    tokens_per_page > 65536 ||
    total_page_count <= 0 ||
    total_page_count > 1073741824 {
    raise InvalidBootstrapSource(KvLayout)
  }
  {
    compute_major,
    compute_minor,
    supports_bf16,
    supports_cublas_lt,
    tokens_per_page,
    total_page_count,
  }
}

///|
pub fn WorkerBootstrapCeilings::new(
  max_model_config_bytes~ : Int,
  max_weight_file_bytes~ : Int64,
  max_weight_arena_bytes~ : Int64,
  max_activation_arena_bytes~ : Int64,
  max_kv_arena_bytes~ : Int64,
  max_execution_manifest_bytes~ : Int64,
  max_module_bytes~ : Int64,
  max_total_module_bytes~ : Int64,
) -> WorkerBootstrapCeilings raise WorkerWireError {
  if max_model_config_bytes <= 0 ||
    max_model_config_bytes > BOOTSTRAP_SOURCE_MAX_MODEL_CONFIG_BYTES ||
    max_weight_file_bytes <= 8L ||
    max_weight_file_bytes > BOOTSTRAP_SOURCE_MAX_DEVICE_BYTES ||
    max_weight_arena_bytes <= 0L ||
    max_weight_arena_bytes > BOOTSTRAP_SOURCE_MAX_DEVICE_BYTES ||
    max_activation_arena_bytes <= 0L ||
    max_activation_arena_bytes > BOOTSTRAP_SOURCE_MAX_DEVICE_BYTES ||
    max_kv_arena_bytes <= 0L ||
    max_kv_arena_bytes > BOOTSTRAP_SOURCE_MAX_DEVICE_BYTES ||
    max_execution_manifest_bytes <= 0L ||
    max_execution_manifest_bytes > BOOTSTRAP_SOURCE_MAX_MANIFEST_BYTES ||
    max_module_bytes <= 0L ||
    max_module_bytes > BOOTSTRAP_SOURCE_MAX_MODULE_BYTES ||
    max_total_module_bytes <= 0L ||
    max_total_module_bytes > BOOTSTRAP_SOURCE_MAX_MODULE_BYTES ||
    max_module_bytes > max_total_module_bytes {
    raise InvalidBootstrapSource(Ceiling)
  }
  {
    max_model_config_bytes,
    max_weight_file_bytes,
    max_weight_arena_bytes,
    max_activation_arena_bytes,
    max_kv_arena_bytes,
    max_execution_manifest_bytes,
    max_module_bytes,
    max_total_module_bytes,
  }
}
