///|
fn bind_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn record_value(document : String, key : String) -> String? {
  let prefix = key + "="
  let mut found : String? = None
  for line in document.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) {
        return None
      }
      found = Some(line[prefix.length():].to_owned())
    }
  }
  found
}

///|
fn require_record_value(
  document : String,
  key : String,
) -> String raise Qwen3Bf16ReleaseBindError {
  guard record_value(document, key) is Some(value) else {
    raise CompiledSetAdmission("missing or duplicate field: " + key)
  }
  value
}

///|
fn require_sha256(
  value : String,
  message : String,
) -> Unit raise Qwen3Bf16ReleaseBindError {
  if !is_lower_sha256(value) {
    raise CompiledSetAdmission(message)
  }
}

///|
fn operation_row(
  document : String,
  operation_id : Int,
) -> FixedArray[String] raise Qwen3Bf16ReleaseBindError {
  let prefix = "operation=\{operation_id},"
  let mut found : FixedArray[String]? = None
  for line in document.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) {
        raise CompiledSetAdmission("duplicate compiled operation")
      }
      found = Some(
        FixedArray::from_array(
          line[10:].split(",").map(value => value.to_owned()).to_array(),
        ),
      )
    }
  }
  guard found is Some(fields) && fields.length() == 5 else {
    raise CompiledSetAdmission("compiled operation row is malformed")
  }
  fields
}

///|
fn read_root_file(
  root : @approved_fs.ApprovedRoot,
  relative : String,
  maximum : Int64,
) -> Bytes raise Qwen3Bf16ReleaseBindError {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(relative)) catch {
    error => {
      ignore(error)
      raise CompiledSetAdmission("compiled-set descendant is unavailable")
    }
  }
  let result : Result[Bytes, Qwen3Bf16ReleaseBindError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum),
  ) catch {
    error => {
      ignore(error)
      Err(CompiledSetAdmission("compiled-set descendant read failed"))
    }
  }
  let close_result : Result[Unit, Qwen3Bf16ReleaseBindError] = Ok(file.close()) catch {
    error => {
      ignore(error)
      Err(CompiledSetAdmission("compiled-set descendant close failed"))
    }
  }
  match (result, close_result) {
    (Ok(bytes), Ok(_)) => bytes
    (Err(error), _) | (_, Err(error)) => raise error
  }
}

///|
async fn approved_driver_identity(
  arguments : Qwen3ReleaseBindArguments,
) -> String raise Qwen3Bf16ReleaseBindError {
  let bytes = @fs.read_file(arguments.toolchain_manifest).binary() catch {
    error => {
      ignore(error)
      raise CompiledSetAdmission("toolchain manifest is unavailable")
    }
  }
  if bind_sha256(bytes) != arguments.toolchain_sha256 {
    raise CompiledSetAdmission("toolchain manifest digest mismatch")
  }
  let text = @utf8.decode(bytes) catch {
    error => {
      ignore(error)
      raise CompiledSetAdmission("toolchain manifest is not UTF-8")
    }
  }
  if require_record_value(text, "schema") !=
    "lunaflux-approved-complete-cuda-toolchain-v1" {
    raise CompiledSetAdmission("toolchain manifest schema mismatch")
  }
  let driver = require_record_value(text, "driver_identity_sha256")
  require_sha256(driver, "driver identity digest is invalid")
  driver
}

///|
async fn admit_compiled_set(
  arguments : Qwen3ReleaseBindArguments,
  candidates : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
) -> Array[@luna_bf16_kernel_producer.LunaBf16OfflineCompilation] raise Qwen3Bf16ReleaseBindError {
  let root = @approved_fs.ApprovedRoot::open_absolute(arguments.compiled_root) catch {
    error => {
      ignore(error)
      raise CompiledSetAdmission("compiled root is unavailable")
    }
  }
  let result : Result[
    Array[@luna_bf16_kernel_producer.LunaBf16OfflineCompilation],
    Qwen3Bf16ReleaseBindError,
  ] = Ok(
    {
      let expected_driver = approved_driver_identity(arguments)
      let compiled_bytes = read_root_file(root, "compiled-set.v1", 1048576L)
      let compiled = @utf8.decode(compiled_bytes) catch {
        error => {
          ignore(error)
          raise CompiledSetAdmission("compiled-set record is not UTF-8")
        }
      }
      if require_record_value(compiled, "schema") !=
        "lunaflux-bf16-compiled-kernel-set.v1" ||
        require_record_value(compiled, "candidate_set_sha256") !=
        candidates.candidate_set_sha256() ||
        require_record_value(compiled, "candidate_inventory_sha256") !=
        candidates.inventory_sha256() ||
        require_record_value(compiled, "toolchain_sha256") !=
        arguments.toolchain_sha256 ||
        require_record_value(compiled, "driver_identity_sha256") !=
        expected_driver ||
        require_record_value(compiled, "target") != "sm_120" ||
        require_record_value(compiled, "operation_count") !=
        candidates.operation_count().to_string() {
        raise CompiledSetAdmission("compiled-set identity mismatch")
      }
      let admitted : Array[
        @luna_bf16_kernel_producer.LunaBf16OfflineCompilation,
      ] = []
      for index, candidate in candidates.candidates() {
        let row = operation_row(compiled, index)
        guard candidates.candidate_key(index) is Some(key) else {
          raise CompiledSetAdmission("candidate key is missing")
        }
        if row[0] != index.to_string() ||
          row[1] != key ||
          row[2] != candidate.family_name() {
          raise CompiledSetAdmission("compiled operation identity mismatch")
        }
        let module_sha = row[3]
        let receipt_sha = row[4]
        require_sha256(module_sha, "compiled module digest is invalid")
        require_sha256(receipt_sha, "compile receipt digest is invalid")
        let receipt_bytes = read_root_file(
          root,
          "receipts/" + key + ".receipt.v1",
          65536L,
        )
        if bind_sha256(receipt_bytes) != receipt_sha {
          raise CompiledSetAdmission("compile receipt digest mismatch")
        }
        let receipt = @utf8.decode(receipt_bytes) catch {
          error => {
            ignore(error)
            raise CompiledSetAdmission("compile receipt is not UTF-8")
          }
        }
        let first_sha = require_record_value(receipt, "first_artifact_sha256")
        let second_sha = require_record_value(receipt, "second_artifact_sha256")
        if require_record_value(receipt, "schema") !=
          "lunaflux-bf16-offline-compile-receipt.v1" ||
          require_record_value(receipt, "candidate_set_sha256") !=
          candidates.candidate_set_sha256() ||
          require_record_value(receipt, "operation_id") != index.to_string() ||
          require_record_value(receipt, "family") != candidate.family_name() ||
          require_record_value(receipt, "source_sha256") !=
          candidate.source_sha256() ||
          require_record_value(receipt, "recipe_sha256") !=
          candidate.recipe_sha256() ||
          require_record_value(receipt, "toolchain_sha256") !=
          arguments.toolchain_sha256 ||
          require_record_value(receipt, "driver_identity_sha256") !=
          expected_driver ||
          require_record_value(receipt, "target") != "sm_120" ||
          first_sha != module_sha ||
          second_sha != module_sha ||
          require_record_value(receipt, "module_relative_path") !=
          "sha256/" + module_sha + ".cubin" ||
          require_record_value(receipt, "deterministic") != "1" {
          raise CompiledSetAdmission("compile receipt binding mismatch")
        }
        let module_bytes = read_root_file(
          root,
          "sha256/" + module_sha + ".cubin",
          4194304L,
        )
        if bind_sha256(module_bytes) != module_sha {
          raise CompiledSetAdmission("compiled module digest mismatch")
        }
        admitted.push(
          @luna_bf16_kernel_producer.LunaBf16OfflineCompilation::new(
            candidate~,
            first_module=module_bytes,
            second_module=module_bytes,
            receipt=@luna_kernel_bundle.LunaDeterministicCompileReceipt::from_offline_builder(
              source_digest_sha256=candidate.source_sha256(),
              recipe_digest_sha256=candidate.recipe_sha256(),
              toolchain_digest_sha256=arguments.toolchain_sha256,
              driver_identity_sha256=expected_driver,
              first_artifact_digest_sha256=first_sha,
              second_artifact_digest_sha256=second_sha,
            ),
          ),
        )
      }
      admitted
    },
  ) catch {
    error => {
      ignore(error)
      Err(CompiledSetAdmission("compiled-set typed admission failed"))
    }
  }
  let close_result : Result[Unit, Qwen3Bf16ReleaseBindError] = Ok(root.close()) catch {
    error => {
      ignore(error)
      Err(CompiledSetAdmission("compiled root close failed"))
    }
  }
  match (result, close_result) {
    (Ok(value), Ok(_)) => value
    (Err(error), _) | (_, Err(error)) => raise error
  }
}
