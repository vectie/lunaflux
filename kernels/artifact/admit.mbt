///|
fn required_entry_points(
  contracts : @launch_contract.LaunchContractSet,
  maximum : Int,
) -> Array[@catalog.AotKernelEntryPoint] raise KernelArtifactError {
  let required : Array[@catalog.AotKernelEntryPoint] = []
  for contract in contracts.contracts() {
    let identity = contract.entry_point()
    if !required.contains(identity) {
      if required.length() >= maximum {
        raise EntryPointCountExceeded(maximum~)
      }
      required.push(identity)
    }
  }
  required
}

///|
fn required_paged_entry_points(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  maximum : Int,
) -> Array[@catalog.AotKernelEntryPoint] raise KernelArtifactError {
  let required : Array[@catalog.AotKernelEntryPoint] = []
  for contract in contracts.contracts() {
    let identity = contract.entry_point()
    if !required.contains(identity) {
      if required.length() >= maximum {
        raise EntryPointCountExceeded(maximum~)
      }
      required.push(identity)
    }
  }
  required
}

///|
fn required_modules(
  entry_points : ArrayView[@catalog.AotKernelEntryPoint],
  maximum : Int,
) -> Array[@catalog.AotArtifactDigest] raise KernelArtifactError {
  let required : Array[@catalog.AotArtifactDigest] = []
  for entry_point in entry_points {
    let digest = entry_point.family().module_digest()
    if !required.contains(digest) {
      if required.length() >= maximum {
        raise ModuleCountExceeded(maximum~)
      }
      required.push(digest)
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
  entry_point : @catalog.AotKernelEntryPoint,
  symbol : String,
  maximum : Int,
) -> Unit raise KernelArtifactError {
  if symbol.is_empty() {
    raise InvalidFunctionSymbol(entry_point~, rule=Empty)
  }
  if symbol.length() > maximum {
    raise InvalidFunctionSymbol(entry_point~, rule=TooLong)
  }
  let mut index = 0
  for character in symbol {
    if character.to_int() > 127 {
      raise InvalidFunctionSymbol(entry_point~, rule=NonAscii)
    }
    if index == 0 {
      if !valid_initial(character) {
        raise InvalidFunctionSymbol(entry_point~, rule=InvalidInitial)
      }
    } else if !valid_continuation(character) {
      raise InvalidFunctionSymbol(entry_point~, rule=InvalidContinuation)
    }
    index += 1
  }
}

///|
fn exact_digest(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn module_input_index(
  inputs : ArrayView[KernelModuleInput],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, input in inputs {
    if module_input_digest(input) == digest {
      return Some(index)
    }
  }
  None
}

///|
fn module_input_digest(input : KernelModuleInput) -> @catalog.AotArtifactDigest {
  match input {
    AliasedModule(digest, _) | OwnedModule(digest, _) => digest
  }
}

///|
fn module_input_length(input : KernelModuleInput) -> Int64 {
  match input {
    AliasedModule(_, bytes) => bytes.length().to_int64()
    OwnedModule(_, bytes) => bytes.length().to_int64()
  }
}

///|
fn own_module_input(input : KernelModuleInput) -> Bytes {
  match input {
    AliasedModule(_, bytes) => Bytes::from_array(bytes)
    OwnedModule(_, bytes) => bytes
  }
}

///|
fn entry_input_index(
  inputs : ArrayView[KernelEntryPointInput],
  identity : @catalog.AotKernelEntryPoint,
) -> Int? {
  for index, input in inputs {
    if input.entry_point == identity {
      return Some(index)
    }
  }
  None
}

///|
fn digest_index(
  digests : ArrayView[@catalog.AotArtifactDigest],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, candidate in digests {
    if candidate == digest {
      return Some(index)
    }
  }
  None
}

///|
fn validate_module_inputs(
  inputs : ArrayView[KernelModuleInput],
  required : ArrayView[@catalog.AotArtifactDigest],
  limits : KernelArtifactLimits,
) -> Int64 raise KernelArtifactError {
  if inputs.length() > limits.max_modules {
    raise ModuleCountExceeded(maximum=limits.max_modules)
  }
  let mut total = 0L
  for index, input in inputs {
    let byte_count = module_input_length(input)
    let digest = module_input_digest(input)
    if byte_count == 0L {
      raise EmptyModule(digest)
    }
    if byte_count > limits.max_module_bytes {
      raise ModuleTooLarge(digest~, maximum=limits.max_module_bytes)
    }
    if total > limits.max_total_module_bytes - byte_count {
      raise TotalModuleBytesExceeded(maximum=limits.max_total_module_bytes)
    }
    total += byte_count
    for earlier in 0..<index {
      if module_input_digest(inputs[earlier]) == digest {
        raise DuplicateModule(digest)
      }
    }
    if !required.contains(digest) {
      raise UnreferencedModule(digest)
    }
  }
  for digest in required {
    if module_input_index(inputs, digest) is None {
      raise MissingModule(digest)
    }
  }
  total
}

///|
fn validate_entry_inputs(
  inputs : ArrayView[KernelEntryPointInput],
  required : ArrayView[@catalog.AotKernelEntryPoint],
  limits : KernelArtifactLimits,
) -> Unit raise KernelArtifactError {
  if inputs.length() > limits.max_entry_points {
    raise EntryPointCountExceeded(maximum=limits.max_entry_points)
  }
  for index, input in inputs {
    validate_symbol(
      input.entry_point,
      input.function_symbol,
      limits.max_function_symbol_bytes,
    )
    for earlier in 0..<index {
      if inputs[earlier].entry_point == input.entry_point {
        raise DuplicateEntryPoint(input.entry_point)
      }
    }
    if !required.contains(input.entry_point) {
      raise UnreferencedEntryPoint(input.entry_point)
    }
  }
  for entry_point in required {
    if entry_input_index(inputs, entry_point) is None {
      raise MissingEntryPoint(entry_point)
    }
  }
}

///|
fn admit_required(
  model_identity : @spec.ModelIdentity,
  device_target : @catalog.DeviceTarget,
  catalog_version : @catalog.CatalogVersion,
  required_entries : Array[@catalog.AotKernelEntryPoint],
  module_inputs : ArrayView[KernelModuleInput],
  entry_point_inputs : ArrayView[KernelEntryPointInput],
  limits : KernelArtifactLimits,
) -> KernelArtifactBundle raise KernelArtifactError {
  let required_digests = required_modules(required_entries, limits.max_modules)
  let total = validate_module_inputs(module_inputs, required_digests, limits)
  validate_entry_inputs(entry_point_inputs, required_entries, limits)
  let modules : Array[KernelModule] = []
  for digest in required_digests {
    guard module_input_index(module_inputs, digest) is Some(input_index) else {
      raise MissingModule(digest)
    }
    let owned_module = own_module_input(module_inputs[input_index])
    if exact_digest(owned_module) != digest.sha256() {
      raise DigestMismatch(digest)
    }
    modules.push({ digest, module_bytes: owned_module })
  }
  let exports : Array[KernelEntryPoint] = []
  for identity in required_entries {
    guard entry_input_index(entry_point_inputs, identity) is Some(input_index) else {
      raise MissingEntryPoint(identity)
    }
    let digest = identity.family().module_digest()
    guard digest_index(required_digests, digest) is Some(module_index) else {
      raise MissingModule(digest)
    }
    exports.push({
      entry_point: identity,
      module_index,
      function_symbol: entry_point_inputs[input_index].function_symbol,
    })
  }
  {
    model_identity,
    device_target,
    catalog_version,
    modules: ReadOnlyArray::from_array(modules),
    entry_points: ReadOnlyArray::from_array(exports),
    total_module_bytes: total,
  }
}

///|
fn ensure_paged_catalog_version(
  actual : @catalog.CatalogVersion,
) -> Unit raise KernelArtifactError {
  let expected = @catalog.CatalogVersion::v3()
  if actual != expected {
    raise CatalogVersionMismatch(expected~, actual~)
  }
}

///|
/// Admits exactly the precompiled modules and CUDA symbols required by an
/// admitted launch-contract set.
///
/// A module is hashed and retained once even when it exposes multiple kernel
/// families or profile-specific entry points. Missing, duplicate,
/// unreferenced, and invalid declarations fail before device import. Distinct
/// stable identities may intentionally alias the same CUDA symbol.
/// No path lookup, executable metadata evaluation, compilation, cache, or JIT
/// is available here.
pub fn admit(
  contracts : @launch_contract.LaunchContractSet,
  module_inputs : ArrayView[KernelModuleInput],
  entry_point_inputs : ArrayView[KernelEntryPointInput],
  limits : KernelArtifactLimits,
) -> KernelArtifactBundle raise KernelArtifactError {
  let required_entries = required_entry_points(
    contracts,
    limits.max_entry_points,
  )
  admit_required(
    contracts.model_identity(),
    contracts.device_target(),
    contracts.catalog_version(),
    required_entries,
    module_inputs,
    entry_point_inputs,
    limits,
  )
}

///|
/// Admits exactly the content-addressed modules and CUDA symbols referenced by
/// a catalog-v3 paged-KV launch-contract set.
///
/// Repeated layer/profile contracts share the first occurrence of an exact
/// stable entry-point identity. Distinct identities may intentionally alias
/// the same bounded CUDA symbol. The result remains inert admission evidence;
/// it does not claim that a worker step or model graph can execute.
pub fn admit_paged_kv(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  module_inputs : ArrayView[KernelModuleInput],
  entry_point_inputs : ArrayView[KernelEntryPointInput],
  limits : KernelArtifactLimits,
) -> KernelArtifactBundle raise KernelArtifactError {
  ensure_paged_catalog_version(contracts.catalog_version())
  let required_entries = required_paged_entry_points(
    contracts,
    limits.max_entry_points,
  )
  admit_required(
    contracts.model_identity(),
    contracts.device_target(),
    contracts.catalog_version(),
    required_entries,
    module_inputs,
    entry_point_inputs,
    limits,
  )
}
