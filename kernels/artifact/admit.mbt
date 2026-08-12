///|
fn required_aot_digests(
  resolved : @catalog.ResolvedKernelCatalog,
  maximum : Int,
) -> Array[@catalog.AotArtifactDigest] raise KernelArtifactError {
  let required : Array[@catalog.AotArtifactDigest] = []
  let seen : Map[String, Unit] = Map([])
  for binding in resolved.bindings() {
    match binding.implementation() {
      VendorBf16GemmImplicitHeuristicV1 => ()
      ContentAddressedAot(digest) => {
        let key = digest.sha256()
        if !seen.contains(key) {
          if required.length() >= maximum {
            raise ArtifactCountExceeded(maximum~)
          }
          seen[key] = ()
          required.push(digest)
        }
      }
    }
  }
  required
}

///|
fn valid_initial(character : Char) -> Bool {
  character is ('A'..='Z' | 'a'..='z' | '_')
}

///|
fn valid_continuation(character : Char) -> Bool {
  valid_initial(character) || character is ('0'..='9')
}

///|
fn validate_symbol(
  digest : @catalog.AotArtifactDigest,
  symbol : String,
  maximum : Int,
) -> Unit raise KernelArtifactError {
  if symbol.is_empty() {
    raise InvalidFunctionSymbol(digest~, rule=Empty)
  }
  if symbol.length() > maximum {
    raise InvalidFunctionSymbol(digest~, rule=TooLong)
  }
  let mut index = 0
  for character in symbol {
    if character.to_int() > 127 {
      raise InvalidFunctionSymbol(digest~, rule=NonAscii)
    }
    if index == 0 {
      if !valid_initial(character) {
        raise InvalidFunctionSymbol(digest~, rule=InvalidInitial)
      }
    } else if !valid_continuation(character) {
      raise InvalidFunctionSymbol(digest~, rule=InvalidContinuation)
    }
    index += 1
  }
}

///|
fn exact_digest(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn validate_inputs(
  inputs : ArrayView[KernelArtifactInput],
  limits : KernelArtifactLimits,
) -> (Map[String, Int], Int64) raise KernelArtifactError {
  if inputs.length() > limits.max_artifacts {
    raise ArtifactCountExceeded(maximum=limits.max_artifacts)
  }
  let by_digest : Map[String, Int] = Map([])
  let mut total = 0L
  for index, input in inputs {
    let byte_count = input.module_bytes.length().to_int64()
    if byte_count == 0L {
      raise EmptyModule(input.digest)
    }
    if byte_count > limits.max_module_bytes {
      raise ModuleTooLarge(digest=input.digest, maximum=limits.max_module_bytes)
    }
    if total > limits.max_total_module_bytes - byte_count {
      raise TotalModuleBytesExceeded(maximum=limits.max_total_module_bytes)
    }
    total += byte_count
    validate_symbol(
      input.digest,
      input.function_symbol,
      limits.max_function_symbol_bytes,
    )
    let key = input.digest.sha256()
    if by_digest.contains(key) {
      raise DuplicateArtifact(input.digest)
    }
    by_digest[key] = index
  }
  (by_digest, total)
}

///|
fn validate_catalog_relationship(
  required : ArrayView[@catalog.AotArtifactDigest],
  inputs : ArrayView[KernelArtifactInput],
  by_digest : Map[String, Int],
) -> Unit raise KernelArtifactError {
  let required_set : Map[String, Unit] = Map([])
  for digest in required {
    let key = digest.sha256()
    required_set[key] = ()
    if !by_digest.contains(key) {
      raise MissingArtifact(digest)
    }
  }
  for input in inputs {
    if !required_set.contains(input.digest.sha256()) {
      raise UnreferencedArtifact(input.digest)
    }
  }
}

///|
/// Admits the exact bounded AOT modules required by a resolved kernel catalog.
///
/// Admission is deliberately separate from device import. It accepts only
/// already-compiled bytes held in memory, verifies SHA-256 over those exact
/// bytes, and rejects duplicate, missing, and unreferenced artifacts. All
/// structural, aggregate, digest, and catalog checks complete before the
/// returned immutable bundle copies any module input. Output order follows
/// first resolved use and is independent of input order. The bundle also pins
/// the resolved model identity, device target, and catalog version.
///
/// There is no path lookup, executable metadata interpretation, compiler,
/// developer cache, or runtime JIT path in this package.
pub fn admit(
  resolved : @catalog.ResolvedKernelCatalog,
  inputs : ArrayView[KernelArtifactInput],
  limits : KernelArtifactLimits,
) -> KernelArtifactBundle raise KernelArtifactError {
  let required = required_aot_digests(resolved, limits.max_artifacts)
  let (by_digest, total) = validate_inputs(inputs, limits)
  validate_catalog_relationship(required, inputs, by_digest)
  let entries : Array[KernelArtifactEntry] = []
  for digest in required {
    guard by_digest.get(digest.sha256()) is Some(input_index) else {
      raise MissingArtifact(digest)
    }
    let input = inputs[input_index]
    let owned_module = Bytes::from_array(input.module_bytes)
    if exact_digest(owned_module) != input.digest.sha256() {
      raise DigestMismatch(input.digest)
    }
    entries.push({
      digest: input.digest,
      module_bytes: owned_module,
      function_symbol: input.function_symbol,
    })
  }
  {
    model_identity: resolved.model_identity(),
    device_target: resolved.device_target(),
    catalog_version: resolved.catalog_version(),
    entries: ReadOnlyArray::from_array(entries),
    total_module_bytes: total,
  }
}
