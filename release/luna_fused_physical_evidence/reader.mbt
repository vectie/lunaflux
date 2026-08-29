///|
priv struct FusedPhysicalReader {
  lines : Array[String]
  mut cursor : Int
}

///|
fn fused_physical_reader(
  canonical : Bytes,
) -> FusedPhysicalReader raise FusedPhysicalEvidenceError {
  if canonical.length() == 0 ||
    canonical.length() > FUSED_PHYSICAL_MAXIMUM_CANONICAL_BYTES {
    raise FusedPhysicalEvidenceLimitExceeded
  }
  let document = @utf8.decode(canonical) catch {
    _ => raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() != FUSED_PHYSICAL_CANONICAL_LINE_COUNT ||
    lines[lines.length() - 1] != "" {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  { lines, cursor: 0 }
}

///|
fn FusedPhysicalReader::read_value(
  self : FusedPhysicalReader,
  key : String,
) -> String raise FusedPhysicalEvidenceError {
  if self.cursor >= self.lines.length() - 1 {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  let prefix = key + "="
  let line = self.lines[self.cursor]
  self.cursor += 1
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  line[prefix.length():].to_owned()
}

///|
fn FusedPhysicalReader::require_literal(
  self : FusedPhysicalReader,
  key : String,
  expected : String,
  rule : FusedPhysicalEvidenceRule,
) -> Unit raise FusedPhysicalEvidenceError {
  if self.read_value(key) != expected {
    raise FusedPhysicalEvidenceMismatch(rule)
  }
}

///|
fn FusedPhysicalReader::read_int(
  self : FusedPhysicalReader,
  key : String,
) -> Int raise FusedPhysicalEvidenceError {
  let text = self.read_value(key)
  let value : Int = @strconv.from_str(text) catch {
    _ => raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  if value.to_string() != text {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  value
}

///|
fn FusedPhysicalReader::read_int64(
  self : FusedPhysicalReader,
  key : String,
) -> Int64 raise FusedPhysicalEvidenceError {
  let text = self.read_value(key)
  let value : Int64 = @strconv.from_str(text) catch {
    _ => raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  if value.to_string() != text {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  value
}

///|
fn FusedPhysicalReader::read_uint(
  self : FusedPhysicalReader,
  key : String,
) -> UInt raise FusedPhysicalEvidenceError {
  let text = self.read_value(key)
  let value : UInt = @strconv.from_str(text) catch {
    _ => raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  if value.to_string() != text {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
  value
}

///|
fn FusedPhysicalReader::require_int(
  self : FusedPhysicalReader,
  key : String,
  expected : Int,
  rule : FusedPhysicalEvidenceRule,
) -> Unit raise FusedPhysicalEvidenceError {
  if self.read_int(key) != expected {
    raise FusedPhysicalEvidenceMismatch(rule)
  }
}

///|
fn FusedPhysicalReader::require_int64(
  self : FusedPhysicalReader,
  key : String,
  expected : Int64,
  rule : FusedPhysicalEvidenceRule,
) -> Unit raise FusedPhysicalEvidenceError {
  if self.read_int64(key) != expected {
    raise FusedPhysicalEvidenceMismatch(rule)
  }
}

///|
fn FusedPhysicalReader::require_uint(
  self : FusedPhysicalReader,
  key : String,
  expected : UInt,
  rule : FusedPhysicalEvidenceRule,
) -> Unit raise FusedPhysicalEvidenceError {
  if self.read_uint(key) != expected {
    raise FusedPhysicalEvidenceMismatch(rule)
  }
}

///|
fn FusedPhysicalReader::read_digest(
  self : FusedPhysicalReader,
  key : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise FusedPhysicalEvidenceError {
  fused_physical_nonzero_digest(self.read_value(key))
}

///|
fn FusedPhysicalReader::require_digest(
  self : FusedPhysicalReader,
  key : String,
  expected : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  rule : FusedPhysicalEvidenceRule,
) -> Unit raise FusedPhysicalEvidenceError {
  if self.read_digest(key) != expected {
    raise FusedPhysicalEvidenceMismatch(rule)
  }
}

///|
fn FusedPhysicalReader::finish(
  self : FusedPhysicalReader,
) -> Unit raise FusedPhysicalEvidenceError {
  if self.cursor != self.lines.length() - 1 {
    raise FusedPhysicalEvidenceInvalid(CanonicalDocument)
  }
}
