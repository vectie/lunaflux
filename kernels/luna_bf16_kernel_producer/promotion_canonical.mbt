///|
fn promotion_append(
  output : StringBuilder,
  written : Ref[Int],
  value : String,
  maximum : Int,
) -> Unit raise LunaBf16PromotionError {
  if value.length() >= maximum || written.val > maximum - value.length() - 1 {
    raise LunaBf16PromotionRejected(CanonicalBytes)
  }
  output.write_string(value)
  output.write_char('\n')
  written.val += value.length() + 1
}

///|
fn promotion_compiler_digest(
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> LunaBf16PromotionDigest {
  let identity = compiler.identity()
  let bytes = @utf8.encode(
    "schema=lunaflux-bf16-compiler-policy.v1\n" +
    "toolchain=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fma=" +
    compiler.fused_multiply_add().to_string() +
    "\n" +
    "reassociate=" +
    compiler.reassociate().to_string() +
    "\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n",
  )
  LunaBf16PromotionDigest(sha256_bytes(bytes))
}

///|
fn append_promotion_shape(
  output : StringBuilder,
  shape : LunaBf16BenchmarkShape,
  maximum : Int,
  written : Ref[Int],
) -> Unit raise LunaBf16PromotionError {
  promotion_append(output, written, shape.prefill_rows.to_string(), maximum)
  promotion_append(output, written, shape.decode_rows.to_string(), maximum)
  promotion_append(output, written, shape.query_tokens.to_string(), maximum)
  promotion_append(
    output,
    written,
    shape.page_table_entries.to_string(),
    maximum,
  )
  promotion_append(output, written, shape.trace.as_hex(), maximum)
}

///|
fn encode_profiler_evidence(
  candidates : LunaBf16CandidateSet,
  profile : @luna_profile_priority.LunaPagedKernelProfileCapture,
  microbenchmarks : ArrayView[LunaBf16MicrobenchmarkObservation],
  end_to_end : ArrayView[LunaBf16EndToEndObservation],
  limits : LunaBf16ProfilerEvidenceLimits,
) -> Bytes raise LunaBf16PromotionError {
  let output = StringBuilder()
  let written = Ref(0)
  let append = value => {
    promotion_append(output, written, value, limits.maximum_canonical_bytes)
  }
  append("lunaflux.bf16-profiler-evidence.v1")
  append(candidates.model_identity().content().as_hex())
  append(candidates.model_identity().plan().as_hex())
  append(candidates.target().compute_major().to_string())
  append(candidates.target().compute_minor().to_string())
  append(candidates.profile().id().as_int().to_string())
  append(candidates.profile().max_query_rows().to_string())
  append(candidates.profile().max_query_tokens().to_string())
  append(candidates.profile().max_page_table_entries().to_string())
  append(candidates.candidate_set_sha256())
  append(promotion_compiler_digest(candidates.compiler_policy()).as_hex())
  append(profile.digest().as_hex())
  append(profile.workload_digest().as_hex())
  append(profile.profiler_digest().as_hex())
  append(limits.trials_per_shape.to_string())
  append(limits.end_to_end_trials.to_string())
  append(limits.maximum_end_to_end_regression_ppm.to_string())
  append(microbenchmarks.length().to_string())
  for observation in microbenchmarks {
    append(observation.operation.as_int().to_string())
    append(observation.shape_ordinal.to_string())
    append(observation.trial_ordinal.to_string())
    append_promotion_shape(
      output,
      observation.shape,
      limits.maximum_canonical_bytes,
      written,
    )
    append(observation.baseline_duration_ns.to_string())
    append(observation.candidate_duration_ns.to_string())
    append(observation.correctness.as_hex())
    append(observation.dispatch_canary.as_hex())
    append(observation.correctness_passed.to_string())
  }
  append(end_to_end.length().to_string())
  for observation in end_to_end {
    append(observation.trial_ordinal.to_string())
    append(observation.workload.as_hex())
    append(observation.baseline_duration_ns.to_string())
    append(observation.candidate_duration_ns.to_string())
    append(observation.correctness.as_hex())
    append(observation.correctness_passed.to_string())
  }
  @utf8.encode(output.to_string())
}

///|
fn encode_launch_contracts(
  contracts : @launch_contract.PagedKvLaunchContractSet,
  maximum : Int,
) -> Bytes raise LunaBf16PromotionError {
  let output = StringBuilder()
  let written = Ref(0)
  let append = value => promotion_append(output, written, value, maximum)
  append("lunaflux.bf16-promotion-launch-contracts.v1")
  append(contracts.model_identity().content().as_hex())
  append(contracts.model_identity().plan().as_hex())
  append(contracts.catalog_version().as_int().to_string())
  append(contracts.device_target().compute_major().to_string())
  append(contracts.device_target().compute_minor().to_string())
  let layout = contracts.device_kv_layout()
  append(layout.stable_layout_version().to_string())
  append(layout.tokens_per_page().to_string())
  append(layout.total_page_count().to_string())
  append(layout.arena_bytes().to_string())
  append(contracts.contracts().length().to_string())
  for contract in contracts.contracts() {
    append(contract.operation_id().as_int().to_string())
    append(contract.profile().id().as_int().to_string())
    append(contract.profile().max_query_rows().to_string())
    append(contract.profile().max_query_tokens().to_string())
    append(contract.profile().max_page_table_entries().to_string())
    append(contract.capability().as_int().to_string())
    append(contract.semantic_version().as_int().to_string())
    append(contract.kv_layout_version().to_string())
    append(contract.tokens_per_page().to_string())
    append(
      match contract.kv_layer() {
        Some(layer) => layer.as_int().to_string()
        None => "absent"
      },
    )
    append(contract.runtime_inputs().length().to_string())
    for input in contract.runtime_inputs() {
      append(
        match input {
          StepCounts => "step-counts"
          QueryPositions => "query-positions"
          QueryRowOffsets => "query-row-offsets"
          SequenceLengths => "sequence-lengths"
          PageTableOffsets => "page-table-offsets"
          PageTablePageIndices => "page-table-page-indices"
        },
      )
    }
    append(
      match contract.persistent_state() {
        NoPersistentState => "none"
        KvCacheReadWrite(layer) => "kv-rw:" + layer.as_int().to_string()
      },
    )
    append(contract.entry_point().family().module_digest().sha256())
    append(contract.entry_point().family().family_id().as_int().to_string())
    append(contract.entry_point().entry_point_id().as_int().to_string())
    let dimensions = contract.dimensions()
    for
      value in [
        dimensions.grid_x(),
        dimensions.grid_y(),
        dimensions.grid_z(),
        dimensions.block_x(),
        dimensions.block_y(),
        dimensions.block_z(),
        dimensions.shared_memory_bytes(),
      ] {
      append(value.to_string())
    }
    append(contract.operands().length().to_string())
    for operand in contract.operands() {
      match operand.role() {
        ExternalInput(reference) => {
          append("external")
          append(
            match reference {
              TokenIds => "token-ids"
            },
          )
        }
        WeightInput(reference) => {
          append("weight")
          append(reference.as_int().to_string())
        }
        WeightScaleInput(reference, scale) => {
          append("weight-scale")
          append(reference.as_int().to_string())
          append(scale.tensor().as_int().to_string())
        }
        ActivationInput(reference) => {
          append("activation-input")
          append(reference.as_int().to_string())
        }
        RuntimeInput(reference) => {
          append("runtime")
          append(
            match reference {
              StepCounts => "step-counts"
              QueryPositions => "query-positions"
              QueryRowOffsets => "query-row-offsets"
              SequenceLengths => "sequence-lengths"
              PageTableOffsets => "page-table-offsets"
              PageTablePageIndices => "page-table-page-indices"
            },
          )
        }
        PersistentKvComponent(layer, component) => {
          append("persistent-kv")
          append(layer.as_int().to_string())
          append(
            match component {
              Key => "key"
              Value => "value"
            },
          )
        }
        ActivationOutput(reference) => {
          append("activation-output")
          append(reference.as_int().to_string())
        }
        Workspace => append("workspace")
      }
      append(operand.byte_count().to_string())
      append(operand.alignment().to_string())
    }
    append(contract.workspace_bytes().to_string())
    append(contract.workspace_alignment().to_string())
  }
  @utf8.encode(output.to_string())
}
