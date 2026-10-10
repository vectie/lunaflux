///|
// Compatibility identity only: no config/checkpoint authentication or hashing.
fn checkpoint_location(input : String) -> (String, @spec.ContentDigest?) raise {
  match input.split("#label=").to_array() {
    [root] => (root.to_owned(), None)
    [root, label] if !root.is_empty() =>
      (
        root.to_owned(),
        Some(@spec.ContentDigest::from_sha256(label.to_owned())),
      )
    _ => fail("VARIANT_ROOT#label=DECLARED_64_HEX_LABEL")
  }
}

///|
fn inventory_prefix_label(prefix : BytesView) -> @spec.ContentDigest raise {
  guard prefix.length() == 72 else { fail("incomplete inventory label") }
  let offset = if prefix[:6] == b"label:" { 6 } else { 0 }
  guard prefix[offset + 64] == b' ' && prefix[offset + 65] == b' ' else {
    fail("invalid inventory label separator")
  }
  @spec.ContentDigest::from_sha256(@utf8.decode(prefix[offset:offset + 64]))
}

///|
fn inventory_label(
  root : @approved_fs.ApprovedRoot,
  name : String,
) -> @spec.ContentDigest? raise {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name)) catch {
    @approved_fs.ApprovedFsError::Unavailable(File) => return None
    error => raise error
  }
  defer file.close()
  // Only the declared first label is needed. Do not read a whole inventory,
  // config or tensor payload to manufacture a content identity.
  let prefix = FixedArray::make(72, b'\x00')
  file.read_exactly_at(prefix, destination_offset=0, byte_count=72, position=0L)
  Some(inventory_prefix_label(Bytes::from_array(prefix[:])))
}

///|
fn checkpoint_label(
  root : @approved_fs.ApprovedRoot,
  supplied : @spec.ContentDigest?,
  mode : String,
  arguments : ArrayView[String],
) -> @spec.ContentDigest raise {
  if supplied is Some(label) {
    return label
  }
  if @env.get_env_var("LUNA_MODEL_CONTENT_LABEL") is Some(label) {
    return @spec.ContentDigest::from_sha256(label)
  }
  let selected = match mode {
    "encode-text" | "decode-audio" | "decode-video" => Some(arguments[0])
    "inspect-component" | "prepare-component" => Some(arguments[1])
    "run-joint" => Some(arguments[8])
    _ => None
  }
  if selected is Some(name) {
    return inventory_label(root, name).unwrap_or_else(() => {
      fail("missing declared checkpoint inventory")
    })
  }
  for
    name in [
      "DENOISER.inventory", "TEXT.inventory", "AUDIO.sha256", "VIDEO.sha256",
    ] {
    if inventory_label(root, name) is Some(label) {
      return label
    }
  }
  fail(
    "supply ROOT#label=HEX or LUNA_MODEL_CONTENT_LABEL; no configuration hashing fallback",
  )
}
