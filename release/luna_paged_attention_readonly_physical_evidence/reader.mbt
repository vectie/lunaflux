///|
priv struct ReadonlyPhysicalReader {
  lines : Array[String]
  mut cursor : Int
}

///|
fn readonly_physical_reader(
  bytes : Bytes,
  maximum : Int,
) -> ReadonlyPhysicalReader raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if bytes.length() == 0 || bytes.length() > maximum {
    raise PagedAttentionReadOnlyPhysicalEvidenceLimitExceeded
  }
  let document = @utf8.decode(bytes) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
  let lines = document.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() != READONLY_PHYSICAL_RESULT_LINES + 1 ||
    lines[lines.length() - 1] != "" {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
  { lines, cursor: 0 }
}

///|
fn ReadonlyPhysicalReader::value(
  self : ReadonlyPhysicalReader,
  key : String,
) -> String raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if self.cursor >= self.lines.length() - 1 {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
  let prefix = key + "="
  let line = self.lines[self.cursor]
  self.cursor += 1
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
  line[prefix.length():].to_owned()
}

///|
fn ReadonlyPhysicalReader::literal(
  self : ReadonlyPhysicalReader,
  key : String,
  expected : String,
  rule : PagedAttentionReadOnlyPhysicalEvidenceRule,
) -> Unit raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if self.value(key) != expected {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(rule)
  }
}

///|
fn ReadonlyPhysicalReader::digest(
  self : ReadonlyPhysicalReader,
  key : String,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest raise PagedAttentionReadOnlyPhysicalEvidenceError {
  readonly_physical_read_digest(self.value(key))
}

///|
fn ReadonlyPhysicalReader::digest_text(
  self : ReadonlyPhysicalReader,
  key : String,
  expected : String,
  rule : PagedAttentionReadOnlyPhysicalEvidenceRule,
) -> Unit raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if self.digest(key).as_hex() != expected {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(rule)
  }
}

///|
fn ReadonlyPhysicalReader::int(
  self : ReadonlyPhysicalReader,
  key : String,
) -> Int raise PagedAttentionReadOnlyPhysicalEvidenceError {
  let text = self.value(key)
  let value : Int = @strconv.from_str(text) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
  if value.to_string() != text {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
  value
}

///|
fn ReadonlyPhysicalReader::exact_int(
  self : ReadonlyPhysicalReader,
  key : String,
  expected : Int,
  rule : PagedAttentionReadOnlyPhysicalEvidenceRule,
) -> Unit raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if self.int(key) != expected {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(rule)
  }
}

///|
fn ReadonlyPhysicalReader::nonnegative_double(
  self : ReadonlyPhysicalReader,
  key : String,
) -> Double raise PagedAttentionReadOnlyPhysicalEvidenceError {
  let text = self.value(key)
  if text.length() == 0 || text.length() > 32 {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Numerical)
  }
  let value = @string.parse_double(text) catch {
    _ => raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Numerical)
  }
  if value < 0.0 || value != value {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Numerical)
  }
  value
}

///|
fn ReadonlyPhysicalReader::finish(
  self : ReadonlyPhysicalReader,
) -> Unit raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if self.cursor != self.lines.length() - 1 {
    raise PagedAttentionReadOnlyPhysicalEvidenceInvalid(Canonical)
  }
}
