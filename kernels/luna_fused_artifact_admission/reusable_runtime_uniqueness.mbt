///|
fn reusable_runtime_validate_unique_inputs(
  modules : ArrayView[ReusableFusedRuntimeModuleInput],
) -> Unit raise FusedArtifactAdmissionError {
  if modules.is_empty() ||
    modules.length() > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES {
    raise InvalidFusedArtifact(Binding)
  }
  let mut total_bytes = 0
  for index, module_ in modules {
    reusable_runtime_validate_module(
      module_.abi,
      module_.first_operation,
      module_.operation_count,
      module_.symbol,
      module_.launch,
      module_.module_bytes,
    )
    total_bytes += module_.module_bytes.length() +
      (match module_.row_variant {
        Some(v) => v.module_bytes.length()
        None => 0
      })
    if total_bytes > REUSABLE_FUSED_RUNTIME_MAXIMUM_TOTAL_MODULE_BYTES {
      raise InvalidFusedArtifact(Module)
    }
    for prior in 0..<index {
      if modules[prior].abi == module_.abi {
        raise MismatchedFusedArtifact(Family)
      }
      if reusable_runtime_spans_overlap(
          modules[prior].first_operation,
          modules[prior].operation_count,
          module_.first_operation,
          module_.operation_count,
        ) &&
        !reusable_runtime_allows_terminal_qwen_attention_overlap(
          modules[prior].abi,
          modules[prior].first_operation,
          module_.abi,
          module_.first_operation,
        ) {
        raise MismatchedFusedArtifact(Binding)
      }
    }
  }
}

///|
fn reusable_runtime_validate_unique_approved(
  modules : ArrayView[ApprovedReusableFusedRuntimeModule],
) -> Unit raise FusedArtifactAdmissionError {
  if modules.is_empty() ||
    modules.length() > REUSABLE_FUSED_RUNTIME_MAXIMUM_MODULES {
    raise InvalidFusedArtifact(Binding)
  }
  let mut total_bytes = 0
  for index, module_ in modules {
    reusable_runtime_validate_module(
      module_.abi,
      module_.first_operation,
      module_.operation_count,
      module_.symbol,
      module_.launch,
      module_.module_bytes,
    )
    total_bytes += module_.module_bytes.length() +
      (match module_.row_variant {
        Some(v) => v.module_bytes.length()
        None => 0
      })
    if total_bytes > REUSABLE_FUSED_RUNTIME_MAXIMUM_TOTAL_MODULE_BYTES {
      raise InvalidFusedArtifact(Module)
    }
    for prior in 0..<index {
      if reusable_runtime_abi_rank(modules[prior].abi) >=
        reusable_runtime_abi_rank(module_.abi) {
        raise MismatchedFusedArtifact(Family)
      }
      if reusable_runtime_spans_overlap(
          modules[prior].first_operation,
          modules[prior].operation_count,
          module_.first_operation,
          module_.operation_count,
        ) &&
        !reusable_runtime_allows_terminal_qwen_attention_overlap(
          modules[prior].abi,
          modules[prior].first_operation,
          module_.abi,
          module_.first_operation,
        ) {
        raise MismatchedFusedArtifact(Binding)
      }
    }
  }
}
