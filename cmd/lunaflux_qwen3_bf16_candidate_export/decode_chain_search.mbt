///|
fn extract_decode_chain_search(
  arguments : ArrayView[String],
) -> (ReadOnlyArray[String], DecodeChainSearch?) raise Qwen3Bf16CandidateExportCliError {
  let remaining : Array[String] = []
  for index = 0, found = None; index < arguments.length(); {
    if arguments[index] != "--decode-chain-search" {
      remaining.push(arguments[index])
      continue index + 1, found
    }
    guard found is None &&
      arguments[index + 1:] is [rows, history, ids, targets, ..] else {
      raise InvalidArguments
    }
    let candidates = ids
      .split(",")
      .to_array()
      .map(s => parse_positive_decimal(s.to_owned()))
    let objectives = targets
      .split(",")
      .to_array()
      .map(s => parse_positive_decimal(s.to_owned()))
    if candidates.is_empty() ||
      candidates.length() > 64 ||
      objectives.is_empty() ||
      objectives.length() > 16 {
      raise InvalidArguments
    }
    for values in [candidates, objectives] {
      for i, value in values {
        if values[:i].contains(value) {
          raise InvalidArguments
        }
      }
    }
    continue index + 5,
      Some({
        rows: parse_positive_decimal(rows),
        history: parse_positive_decimal(history),
        candidates: ReadOnlyArray::from_array(candidates),
        targets: ReadOnlyArray::from_array(objectives),
      })
  } nobreak {
    (ReadOnlyArray::from_array(remaining), found)
  }
}

///|
/// An offline alternative set, not a change to the ordinary serving module.
/// Shape comes from the model plan; search and compilation are model-neutral.
fn prepare_decode_chain_search(
  search : DecodeChainSearch,
  baseline : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
  target : @catalog.DeviceTarget,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier raise Qwen3Bf16CandidateExportCliError {
  let abi = baseline.source().abi()
  let (maximum_rows, _, _) = abi.limits()
  if search.rows > maximum_rows ||
    search.history > baseline.functional().plan().problem().dimensions().1 {
    raise InvalidArguments
  }
  let request = @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest::new(
    backend=@luna_cuda_attention_autotune.cuda_attention_backend(),
    target_id=@luna_cuda_attention_autotune.cuda_architecture_target(
      target.compute_major(),
      target.compute_minor(),
    ),
    problem=@luna_attention_strategy.AttentionCompileProblem::new(
      shape=baseline.functional().plan().problem().shape(),
      phase=Decode,
      numeric_type=AttentionBFloat16,
      query_rows=1,
      batch_rows=search.rows,
      context_tokens=search.history,
      causal=true,
      allow_alternative_softmax_reduction=true,
    ),
    capabilities=@luna_cuda_attention_autotune.cuda_sm120_decode_compiler_capabilities(),
    device_target=target,
    maximum_shared_memory_bytes=49152L,
    abi~,
    partial_symbol="lunaflux_decode_chain_partial",
    merge_symbol="lunaflux_decode_chain_merge",
    maximum_workspace_bytes=268435456L,
    maximum_source_bytes=1048576,
  ) catch {
    _ => raise QwenDecodeSplitPreparation
  }
  let frontier = @luna_cuda_attention_tile_aot.enumerate_cuda_partitioned_attention_tiles(
    request,
    search.candidates[:],
    search.targets[:],
  ) catch {
    _ => raise QwenDecodeSplitPreparation
  }
  // This export recipe describes the row-direct blockwise/matrix ABI. Do not
  // attach it to a different executable ownership/merge family.
  if !frontier
    .variants()
    .all(v => {
      qwen_decode_physical_numeric_law(v.compilation().lowering()) is Some(_)
    }) {
    raise QwenDecodeSplitPreparation
  }
  frontier
}

///|
fn decode_chain_scope(
  frontier : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier,
  device : String,
  toolchain : String,
  compiler_flags : String,
) -> String {
  frontier.digest().as_hex() +
  "\t" +
  device +
  "\t" +
  toolchain +
  "\t" +
  compiler_flags +
  "\tstandalone-partial-merge-v1"
}

///|
fn decode_chain_compiler_flags(
  policy : @luna_specialization.LunaCompilerPolicy,
) -> String {
  let identity = policy.identity()
  "version=\{identity.major()}.\{identity.minor()}.\{identity.patch()},opt=\{policy.optimization_level()},fmad=\{policy.fused_multiply_add()},reassociate=\{policy.reassociate()},registers=\{policy.max_registers()}"
}

///|
fn select_decode_chain_record(
  frontier : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier,
  search : DecodeChainSearch,
  input : (Bytes, String, String),
  compiler_flags~ : String,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation raise Qwen3Bf16CandidateExportCliError {
  let (bytes, device, toolchain) = input
  if bytes.length() > 1048576 {
    raise InvalidArguments
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidArguments }
  let scope = decode_chain_scope(frontier, device, toolchain, compiler_flags)
  guard text.split("\n").to_array()
    is ["luna-decode-chain-timing-v1", actual_scope, workload, .. lines, ""] &&
    actual_scope == "scope\t" + scope &&
    workload == "workload\tdecode\t\{search.rows}\t\{search.history}" &&
    !lines.is_empty() &&
    lines.length() <= frontier.variants().length() else {
    raise InvalidArguments
  }
  let observations = lines.map(line => {
    guard line.split("\t").to_array() is ["chain", digest, elapsed, samples] else {
      raise InvalidArguments
    }
    guard frontier
      .variants()
      .filter(v => v.compilation().digest().as_hex() == digest.to_owned())
      is [variant] else {
      raise InvalidArguments
    }
    @luna_cuda_attention_tile_aot.PartitionedAttentionChainObservation::new(
      frontier=frontier.digest(),
      compilation=variant.compilation().digest(),
      scope~,
      elapsed_ns=decode_chain_latency(elapsed),
      samples=parse_positive_decimal(samples.to_owned()),
    ) catch {
      _ => raise InvalidArguments
    }
  })
  frontier.select_measured_chain(
    scope~,
    observations[:],
    maximum_workspace_bytes=268435456L,
    maximum_shared_memory_bytes=49152L,
  ) catch {
    _ => raise InvalidArguments
  }
}

///|
async fn publish_decode_chain_search(
  output : String,
  search : DecodeChainSearch,
  frontier : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier,
  selected : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation?,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  operation : Int,
) -> Unit {
  for index, variant in frontier.variants() {
    let compilation = variant.compilation()
    write_new_candidate_pair(
      output + "/decode-chain-v\{index}",
      compilation.source().bytes(),
      decode_chain_recipe(compilation, frontier, search, set, operation),
    )
  }
  if selected is Some(compilation) {
    write_new_candidate_pair(
      output + "/decode-chain-selected",
      compilation.source().bytes(),
      decode_chain_recipe(compilation, frontier, search, set, operation),
    )
    println("decode_chain_selected_sha256=\{compilation.digest().as_hex()}")
  }
  println("decode_chain_frontier_sha256=\{frontier.digest().as_hex()}")
  println("decode_chain_variants=\{frontier.variants().length()}")
}

///|
fn decode_chain_recipe(
  compilation : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation,
  frontier : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier,
  search : DecodeChainSearch,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  operation : Int,
) -> Bytes {
  let recipe = qwen_compiler_partitioned_attention_recipe(
    operation,
    set,
    compilation,
    frontier_sha256=frontier.digest().as_hex(),
  )
  let lowering = compilation.lowering()
  let (gx, gy, block, shared) = lowering.launch()
  let partitions = lowering.key_value_partitions()
  let text = @utf8.decode_lossy(recipe)
    .replace(
      old="family=paged-attention-prefill-functional-partitioned-tile",
      new="family=paged-attention-decode-functional-partitioned-tile",
    )
    .replace(
      old=qwen_compiler_partitioned_launch_record(
        gx, gy, partitions, block, shared,
      ),
      new=decode_chain_launch_record(compilation, search.rows),
    )
  let numeric_law = qwen_decode_physical_numeric_law(lowering).unwrap_or(
    "non-blockwise",
  )
  @utf8.encode(
    text +
    "candidate_id=\{compilation.functional().plan().candidate().stable_id()}\n" +
    "chain_compiler_flags=\{decode_chain_compiler_flags(set.compiler_policy())}\n" +
    "numeric_law=\{numeric_law}\n" +
    "compilation_sha256=\{compilation.digest().as_hex()}\n" +
    "benchmark_batch_rows=\{search.rows}\nbenchmark_history=\{search.history}\n" +
    "measurement_boundary=standalone-partial-merge-v1\nserving_selected=false\n",
  )
}

///|
fn decode_chain_latency(
  text : StringView,
) -> Int64 raise Qwen3Bf16CandidateExportCliError {
  if text.is_empty() || text.length() > 19 {
    raise InvalidArguments
  }
  for unit in text; value = 0L {
    guard unit is ('0'..='9') else { raise InvalidArguments }
    let digit = (unit.to_int() - '0'.to_int()).to_int64()
    if value > (9223372036854775807L - digit) / 10L {
      raise InvalidArguments
    }
    continue value * 10L + digit
  } nobreak {
    if value <= 0L {
      raise InvalidArguments
    }
    value
  }
}

///|
fn decode_chain_launch_record(
  compilation : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation,
  rows : Int,
) -> String {
  let lowering = compilation.lowering()
  let (_, groups, threads, shared) = lowering.launch()
  let (heads, _, _, _) = compilation.source().abi().shape()
  let splits = lowering.key_value_partitions()
  "\npartial_grid=\{rows},\{groups},\{splits}\nmerge_grid=\{rows},\{heads},1" +
  "\nblock=\{threads},1,1\npartial_shared_memory_bytes=\{shared}\nmerge_shared_memory_bytes=0"
}
