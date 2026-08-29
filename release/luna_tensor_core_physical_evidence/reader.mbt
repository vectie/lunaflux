///|
priv struct LunaTensorCoreEvidenceReader {
  lines : Array[String]
  mut cursor : Int
}

///|
fn tensor_core_evidence_reader(
  canonical : Bytes,
  maximum_bytes : Int,
) -> LunaTensorCoreEvidenceReader raise LunaTensorCorePhysicalEvidenceError {
  if canonical.length() == 0 || canonical.length() > maximum_bytes {
    raise LunaTensorCorePhysicalEvidenceLimitExceeded
  }
  let document = @utf8.decode(canonical) catch {
    _ => raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() != TENSOR_CORE_EVIDENCE_RESULT_LINES + 1 ||
    lines[lines.length() - 1] != "" {
    raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  { lines, cursor: 0 }
}

///|
fn LunaTensorCoreEvidenceReader::read_value(
  self : LunaTensorCoreEvidenceReader,
  key : String,
) -> String raise LunaTensorCorePhysicalEvidenceError {
  if self.cursor >= self.lines.length() - 1 {
    raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  let prefix = key + "="
  let line = self.lines[self.cursor]
  self.cursor += 1
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  line[prefix.length():].to_owned()
}

///|
fn LunaTensorCoreEvidenceReader::require_literal(
  self : LunaTensorCoreEvidenceReader,
  key : String,
  expected : String,
  rule : LunaTensorCorePhysicalEvidenceRule,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  if self.read_value(key) != expected {
    raise LunaTensorCorePhysicalEvidenceMismatch(rule)
  }
}

///|
fn LunaTensorCoreEvidenceReader::read_digest(
  self : LunaTensorCoreEvidenceReader,
  key : String,
) -> LunaTensorCorePhysicalEvidenceDigest raise LunaTensorCorePhysicalEvidenceError {
  tensor_core_evidence_read_digest(self.read_value(key))
}

///|
fn LunaTensorCoreEvidenceReader::require_digest_text(
  self : LunaTensorCoreEvidenceReader,
  key : String,
  expected : String,
  rule : LunaTensorCorePhysicalEvidenceRule,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  if self.read_digest(key).as_hex() != expected {
    raise LunaTensorCorePhysicalEvidenceMismatch(rule)
  }
}

///|
fn LunaTensorCoreEvidenceReader::read_int(
  self : LunaTensorCoreEvidenceReader,
  key : String,
) -> Int raise LunaTensorCorePhysicalEvidenceError {
  let text = self.read_value(key)
  let value : Int = @strconv.from_str(text) catch {
    _ => raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  if value.to_string() != text {
    raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  value
}

///|
fn LunaTensorCoreEvidenceReader::require_int(
  self : LunaTensorCoreEvidenceReader,
  key : String,
  expected : Int,
  rule : LunaTensorCorePhysicalEvidenceRule,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  if self.read_int(key) != expected {
    raise LunaTensorCorePhysicalEvidenceMismatch(rule)
  }
}

///|
fn LunaTensorCoreEvidenceReader::read_int64(
  self : LunaTensorCoreEvidenceReader,
  key : String,
) -> Int64 raise LunaTensorCorePhysicalEvidenceError {
  let text = self.read_value(key)
  let value : Int64 = @strconv.from_str(text) catch {
    _ => raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  if value.to_string() != text {
    raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
  value
}

///|
fn LunaTensorCoreEvidenceReader::read_nonnegative_double(
  self : LunaTensorCoreEvidenceReader,
  key : String,
) -> Double raise LunaTensorCorePhysicalEvidenceError {
  let text = self.read_value(key)
  if text.length() == 0 ||
    text.length() > 32 ||
    (text.length() > 1 && text[0] == '0' && text[1] != '.') {
    raise LunaTensorCorePhysicalEvidenceInvalid(Numerical)
  }
  let mut dots = 0
  for character in text {
    if character == '.' {
      dots += 1
    } else if !(character >= '0' && character <= '9') {
      raise LunaTensorCorePhysicalEvidenceInvalid(Numerical)
    }
  }
  if dots > 1 || text[text.length() - 1] == '.' {
    raise LunaTensorCorePhysicalEvidenceInvalid(Numerical)
  }
  let value = @string.parse_double(text) catch {
    _ => raise LunaTensorCorePhysicalEvidenceInvalid(Numerical)
  }
  if value < 0.0 || value != value {
    raise LunaTensorCorePhysicalEvidenceInvalid(Numerical)
  }
  value
}

///|
fn LunaTensorCoreEvidenceReader::finish(
  self : LunaTensorCoreEvidenceReader,
) -> Unit raise LunaTensorCorePhysicalEvidenceError {
  if self.cursor != self.lines.length() - 1 {
    raise LunaTensorCorePhysicalEvidenceInvalid(CanonicalDocument)
  }
}
