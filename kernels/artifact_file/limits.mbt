///|
const ABSOLUTE_MAX_MANIFEST_BYTES : Int64 = 67108864L

///|
const ABSOLUTE_MAX_JSON_DEPTH : Int = 64

///|
const ABSOLUTE_MAX_PATH_BYTES : Int = 4096

///|
/// Complete startup bounds for one manifest and its immutable AOT modules.
///
/// Module counts, module bytes, aggregate bytes, entry-point counts, and CUDA
/// symbol bytes remain owned by the existing semantic artifact limits.
pub struct KernelArtifactFileLimits {
  max_manifest_bytes : Int64
  max_json_depth : Int
  max_path_bytes : Int
  artifact : @artifact.KernelArtifactLimits
}

///|
pub fn KernelArtifactFileLimits::new(
  max_manifest_bytes~ : Int64,
  max_json_depth~ : Int,
  max_path_bytes~ : Int,
  artifact~ : @artifact.KernelArtifactLimits,
) -> KernelArtifactFileLimits raise KernelArtifactFileError {
  if max_manifest_bytes <= 0L ||
    max_manifest_bytes > ABSOLUTE_MAX_MANIFEST_BYTES {
    raise InvalidLimit(ManifestBytes)
  }
  if max_json_depth <= 0 || max_json_depth > ABSOLUTE_MAX_JSON_DEPTH {
    raise InvalidLimit(JsonDepth)
  }
  if max_path_bytes <= 0 || max_path_bytes > ABSOLUTE_MAX_PATH_BYTES {
    raise InvalidLimit(PathBytes)
  }
  { max_manifest_bytes, max_json_depth, max_path_bytes, artifact }
}

///|
pub fn KernelArtifactFileLimits::max_manifest_bytes(
  self : KernelArtifactFileLimits,
) -> Int64 {
  self.max_manifest_bytes
}

///|
pub fn KernelArtifactFileLimits::max_json_depth(
  self : KernelArtifactFileLimits,
) -> Int {
  self.max_json_depth
}

///|
pub fn KernelArtifactFileLimits::max_path_bytes(
  self : KernelArtifactFileLimits,
) -> Int {
  self.max_path_bytes
}

///|
pub fn KernelArtifactFileLimits::artifact(
  self : KernelArtifactFileLimits,
) -> @artifact.KernelArtifactLimits {
  self.artifact
}
