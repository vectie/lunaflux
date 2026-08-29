///|
fn kv_physical_reader(
  canonical : Bytes,
) -> PagedKvWritePhysicalReader raise PagedKvWritePhysicalError {
  if canonical.is_empty() ||
    canonical.length() > PAGED_KV_WRITE_PHYSICAL_MAX_BYTES {
    raise LimitExceeded
  }
  let text = @utf8.decode(canonical) catch { _ => raise Invalid(Canonical) }
  let lines = text.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() < 2 || lines[lines.length() - 1] != "" {
    raise Invalid(Canonical)
  }
  { lines, cursor: 0 }
}

///|
fn PagedKvWritePhysicalReader::value(
  self : PagedKvWritePhysicalReader,
  key : String,
) -> String raise PagedKvWritePhysicalError {
  if self.cursor >= self.lines.length() - 1 {
    raise Invalid(Canonical)
  }
  let prefix = key + "="
  let line = self.lines[self.cursor]
  self.cursor += 1
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise Invalid(Canonical)
  }
  line[prefix.length():].to_owned()
}

///|
fn PagedKvWritePhysicalReader::literal(
  self : PagedKvWritePhysicalReader,
  key : String,
  expected : String,
  rule : PagedKvWritePhysicalRule,
) -> Unit raise PagedKvWritePhysicalError {
  if self.value(key) != expected {
    raise Mismatch(rule)
  }
}

///|
fn PagedKvWritePhysicalReader::int64(
  self : PagedKvWritePhysicalReader,
  key : String,
) -> Int64 raise PagedKvWritePhysicalError {
  let text = self.value(key)
  let value : Int64 = @strconv.from_str(text) catch {
    _ => raise Invalid(Canonical)
  }
  if value.to_string() != text {
    raise Invalid(Canonical)
  }
  value
}

///|
fn PagedKvWritePhysicalReader::digest(
  self : PagedKvWritePhysicalReader,
  key : String,
) -> @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest raise PagedKvWritePhysicalError {
  @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest::from_sha256(self.value(key)) catch {
    _ => raise Invalid(Digest)
  }
}

///|
fn PagedKvWritePhysicalReader::finish(
  self : PagedKvWritePhysicalReader,
) -> Unit raise PagedKvWritePhysicalError {
  if self.cursor != self.lines.length() - 1 {
    raise Invalid(Canonical)
  }
}
