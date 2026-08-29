///|
/// Digest-pinned numeric model source for the production I8 worker recipe.
/// The artifact digest is independent of the model/config identities.
pub struct WorkerI8BootstrapModelSource {
  priv model : WorkerBootstrapModelSource
  priv numeric_weight_artifact_digest : String
}

///|
/// Exact pre-hardware device facts admitted by the production I8 recipe.
pub struct WorkerI8BootstrapExecution {
  priv execution : WorkerBootstrapExecution
  priv total_memory_bytes : Int64
} derive(Debug, Eq)

///|
pub fn WorkerI8BootstrapModelSource::new(
  config_locator~ : StringView,
  numeric_weights_locator~ : StringView,
  config_digest_sha256~ : StringView,
  numeric_weight_artifact_sha256~ : StringView,
) -> WorkerI8BootstrapModelSource raise WorkerWireError {
  if !bootstrap_source_is_sha256(numeric_weight_artifact_sha256) {
    raise InvalidBootstrapSource(Digest)
  }
  {
    model: WorkerBootstrapModelSource::new(
      config_locator~,
      weights_locator=numeric_weights_locator,
      config_digest_sha256~,
    ),
    numeric_weight_artifact_digest: numeric_weight_artifact_sha256.to_owned(),
  }
}

///|
pub fn WorkerI8BootstrapExecution::new(
  compute_major~ : Int,
  compute_minor~ : Int,
  total_memory_bytes~ : Int64,
  supports_bf16~ : Bool,
  supports_cublas_lt~ : Bool,
  tokens_per_page~ : Int,
  total_page_count~ : Int,
) -> WorkerI8BootstrapExecution raise WorkerWireError {
  if !((compute_major == 8 && compute_minor == 9) ||
    (compute_major == 9 && compute_minor == 0) ||
    (compute_major == 12 && compute_minor == 0)) ||
    total_memory_bytes <= 0L ||
    total_memory_bytes > BOOTSTRAP_SOURCE_MAX_DEVICE_BYTES ||
    !supports_bf16 ||
    !supports_cublas_lt {
    raise InvalidBootstrapSource(Target)
  }
  {
    execution: WorkerBootstrapExecution::new(
      compute_major~,
      compute_minor~,
      supports_bf16~,
      supports_cublas_lt~,
      tokens_per_page~,
      total_page_count~,
    ),
    total_memory_bytes,
  }
}

///|
pub fn WorkerI8BootstrapModelSource::config_locator(self : Self) -> String {
  self.model.config_locator()
}

///|
pub fn WorkerI8BootstrapModelSource::numeric_weights_locator(
  self : Self,
) -> String {
  self.model.weights_locator()
}

///|
pub fn WorkerI8BootstrapModelSource::config_digest(self : Self) -> String {
  self.model.config_digest()
}

///|
pub fn WorkerI8BootstrapModelSource::numeric_weight_artifact_digest(
  self : Self,
) -> String {
  self.numeric_weight_artifact_digest
}

///|
pub fn WorkerI8BootstrapExecution::compute_major(self : Self) -> Int {
  self.execution.compute_major()
}

///|
pub fn WorkerI8BootstrapExecution::compute_minor(self : Self) -> Int {
  self.execution.compute_minor()
}

///|
pub fn WorkerI8BootstrapExecution::total_memory_bytes(self : Self) -> Int64 {
  self.total_memory_bytes
}

///|
pub fn WorkerI8BootstrapExecution::supports_bf16(self : Self) -> Bool {
  self.execution.supports_bf16()
}

///|
pub fn WorkerI8BootstrapExecution::supports_cublas_lt(self : Self) -> Bool {
  self.execution.supports_cublas_lt()
}

///|
pub fn WorkerI8BootstrapExecution::tokens_per_page(self : Self) -> Int {
  self.execution.tokens_per_page()
}

///|
pub fn WorkerI8BootstrapExecution::total_page_count(self : Self) -> Int {
  self.execution.total_page_count()
}
