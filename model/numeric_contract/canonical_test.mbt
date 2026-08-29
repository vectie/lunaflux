///|
fn canonical_plain_storage() -> @numeric_contract.TensorStorageContract raise {
  @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::bfloat16(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn canonical_bf16_execution() -> @numeric_contract.OperationExecutionContract raise {
  @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::bfloat16(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn canonical_compound_fp8_execution() -> @numeric_contract.OperationExecutionContract raise {
  @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
test "both canonical record kinds round trip exactly" {
  let storage = canonical_plain_storage()
  assert_eq(storage.canonical_bytes().length(), 45)
  assert_eq(
    storage.digest().as_hex(),
    "aa5910bd7bb7cc6e1ea9e6edf265a4e5141ed7378b7aa98aa71b0acb34914779",
  )
  assert_eq(
    @numeric_contract.load_tensor_storage_contract(
      storage.canonical_bytes(),
      storage.digest(),
    ),
    storage,
  )
  let execution = canonical_bf16_execution()
  assert_eq(execution.canonical_bytes().length(), 51)
  assert_eq(
    execution.digest().as_hex(),
    "eb581d0a38ea6faa1ddaf89db1a33606d18cadd655ad8094c76c7cc7fba60f99",
  )
  assert_eq(
    @numeric_contract.load_operation_execution_contract(
      execution.canonical_bytes(),
      execution.digest(),
    ),
    execution,
  )
}

///|
test "compound gated execution has a pinned canonical digest" {
  let execution = canonical_compound_fp8_execution()
  assert_eq(execution.canonical_bytes().length(), 51)
  inspect(
    execution.digest().as_hex(),
    content="a4b2e26734da392ca0b26af56820da08856fe339523ba3cbe5c757d88a598e17",
  )
  assert_eq(
    try! @numeric_contract.load_operation_execution_contract(
      execution.canonical_bytes(),
      execution.digest(),
    ),
    execution,
  )
}

///|
test "on-demand fixed canonical bytes are stable independent values" {
  let storage = canonical_plain_storage()
  let first = storage.canonical_bytes()
  let second = storage.canonical_bytes()
  assert_eq(first, second)
  assert_eq(
    @crypto.bytes_to_hex_string(@crypto.sha256(first)),
    storage.digest().as_hex(),
  )
  let execution = canonical_bf16_execution()
  assert_eq(execution.canonical_bytes(), execution.canonical_bytes())
}

///|
test "every storage byte is digest-bound" {
  let storage = canonical_plain_storage()
  let canonical = storage.canonical_bytes()
  for changed_index in 0..<canonical.length() {
    let changed = Bytes::makei(canonical.length(), index => {
      if index == changed_index {
        (canonical[index].to_int() ^ 1).to_byte()
      } else {
        canonical[index]
      }
    })
    try
      @numeric_contract.load_tensor_storage_contract(changed, storage.digest())
    catch {
      TensorStorageDigestMismatch => ()
      _ => fail("unexpected storage mutation failure")
    } noraise {
      _ => fail("storage mutation retained its digest")
    }
  }
}

///|
test "operation canonical framing rejects foreign domains" {
  let execution = canonical_bf16_execution()
  let canonical = execution.canonical_bytes()
  let changed = Bytes::makei(canonical.length(), index => {
    if index == 0 {
      (canonical[index].to_int() ^ 1).to_byte()
    } else {
      canonical[index]
    }
  })
  let digest = @numeric_contract.OperationExecutionDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(changed)),
  )
  try
    @numeric_contract.load_operation_execution_contract(changed, digest)
  catch {
    InvalidNumericContract(CanonicalBytes) => ()
    _ => fail("unexpected operation-domain failure")
  } noraise {
    _ => fail("foreign operation domain was admitted")
  }
}

///|
test "operation loader rejects the retired v1 domain without defaults" {
  let legacy : Array[Byte] = []
  for byte in b"lunaflux.operation-execution-contract.v1\x00" {
    legacy.push(byte)
  }
  for tag in [1, 1, 1, 1, 0, 0, 1, 1] {
    legacy.push(tag.to_byte())
  }
  let bytes = Bytes::from_array(legacy)
  let digest = @numeric_contract.OperationExecutionDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  try @numeric_contract.load_operation_execution_contract(bytes, digest) catch {
    InvalidNumericContract(CanonicalBytes) => ()
    _ => fail("unexpected retired operation-domain failure")
  } noraise {
    _ => fail("retired operation v1 bytes were defaulted into v2")
  }
}

///|
test "digest text is canonical lowercase SHA-256" {
  for
    value in [
      "", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde", "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
    try @numeric_contract.TensorStorageDigest::from_sha256(value) catch {
      InvalidNumericContract(Digest) => ()
      _ => fail("unexpected digest failure")
    } noraise {
      _ => fail("noncanonical digest text was admitted")
    }
  }
}

///|
test "operation loader rejects unknown activation-scale policy tags" {
  let execution = canonical_compound_fp8_execution()
  let canonical = execution.canonical_bytes()
  // The policy byte is the third field after the fixed v2 domain.
  let policy_offset = canonical.length() - 8
  let changed = Bytes::makei(canonical.length(), index => {
    if index == policy_offset {
      (4).to_byte()
    } else {
      canonical[index]
    }
  })
  let digest = @numeric_contract.OperationExecutionDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(changed)),
  )
  try
    @numeric_contract.load_operation_execution_contract(changed, digest)
  catch {
    InvalidNumericContract(ExecutionCombination) => ()
    _ => fail("unexpected unknown activation-scale policy failure")
  } noraise {
    _ => fail("unknown activation-scale policy was admitted")
  }
}
