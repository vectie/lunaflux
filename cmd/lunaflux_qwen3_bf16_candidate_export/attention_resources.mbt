///|
fn qwen_attention_resource_request(
  request : @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest,
  frontier : @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier,
  resources : (Bytes, String, String)?,
) -> @luna_cuda_attention_tile_aot.CudaAttentionTileAotRequest raise Qwen3Bf16CandidateExportCliError {
  guard resources is Some((bytes, device, toolchain)) else { return request }
  let scope = frontier.digest().as_hex() + "\t" + device + "\t" + toolchain
  let budget = @luna_attention_tuning.parse_attention_resource_budget(
    bytes,
    scope~,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let expanded_request = request.with_resource_feedback(budget, []) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let expanded = @luna_cuda_attention_tile_aot.compile_cuda_attention_tile_aot_frontier(
    expanded_request,
    [],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let (budget, observations) = @luna_attention_tuning.parse_attention_resource_feedback(
    bytes,
    scope~,
    expanded.variants().map(v => v.functional().schedule().digest())[:],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  request.with_resource_feedback(budget, observations[:]) catch {
    _ => raise QwenReadOnlyPreparation
  }
}

///|
async fn publish_decode_frontier(
  output : String,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  fused : QwenFusedProductionSet,
) -> Unit {
  for variant in fused.attention_decode_frontier.variants() {
    let id = variant.functional().plan().candidate().stable_id()
    write_new_candidate_pair(
      output + "/reusable-qwen-decode-attention-c" + id.to_string(),
      variant.source().bytes(),
      qwen_compiler_attention_recipe(
        fused.first_operation + 3,
        "paged-attention-decode-functional-tile",
        set,
        variant,
        frontier_sha256=fused.attention_decode_frontier.digest().as_hex(),
        variant_stable_id=id,
      ),
    )
  }
}

///|
fn qwen_partitioned_resource_request(
  request : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest,
  frontier : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotFrontier,
  resources : (Bytes, String, String)?,
) -> @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotRequest raise Qwen3Bf16CandidateExportCliError {
  guard resources is Some((bytes, device, toolchain)) else { return request }
  let scope = frontier.digest().as_hex() + "\t" + device + "\t" + toolchain
  let budget = @luna_attention_tuning.parse_attention_resource_budget(
    bytes,
    scope~,
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let expanded = @luna_cuda_attention_tile_aot.compile_cuda_partitioned_attention_tile_aot_frontier(
    request.with_resource_feedback(budget, []),
    [],
    [32, 64, 128],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  let (budget, observations) = @luna_attention_tuning.parse_attention_resource_feedback(
    bytes,
    scope~,
    expanded
    .variants()
    .map(v => v.compilation().functional().schedule().digest())[:],
  ) catch {
    _ => raise QwenReadOnlyPreparation
  }
  request.with_resource_feedback(budget, observations[:]) catch {
    _ => raise QwenReadOnlyPreparation
  }
}
