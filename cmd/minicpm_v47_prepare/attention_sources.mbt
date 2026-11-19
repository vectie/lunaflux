///|
async fn prepare_attention_sources(
  operators : @operators.DecoderOperators,
  attention : @attention.Compilation,
  program : @attention.Program,
  output : String,
) -> Json {
  @fs.write_file(
    output + "/attention_common.cu",
    program.common_source_bytes(),
    create_mode=CreateNew,
  )
  let kernels : Array[Json] = []
  for
    (name, phase, compilation) in [
      ("prefill", @attention.Phase::Prefill, attention.prefill()),
      ("decode", Decode, attention.decode()),
    ] {
    let filename = "attention_" + name + ".cu"
    @fs.write_file(
      output + "/" + filename,
      program.attention_source_bytes(phase),
      create_mode=CreateNew,
    )
    kernels.push({
      "phase": Json(name),
      "source": Json(filename),
      "symbol": Json(compilation.source().symbol().to_owned()),
      "launch": Json(compilation.lowering().launch()),
      "candidate_id": Json(
        compilation.functional().plan().candidate().stable_id(),
      ),
      "selection": "unmeasured legal emitted schedule",
      "source_digest": Json(compilation.source().digest().as_hex()),
      "ordered_stages": Json(
        program
        .stages(phase)
        .map(stage => {
          "symbol": Json(stage.symbol()),
          "source": Json(
            if stage.is_attention() {
              filename
            } else {
              "attention_common.cu"
            },
          ),
          "launch": Json(stage.launch()),
          "arguments": Json(stage.arguments().map(slot => slot.name())),
        }),
      ),
    })
  }
  {
    "common_source": "attention_common.cu",
    "common_entrypoints": 12,
    "ordered_stages_per_phase": 13,
    "workspace_bytes": Json(program.workspace_bytes()),
    "regions": Json(
      program
      .regions()
      .map(region => {
        "slot": Json(region.slot.name()),
        "workspace_offset": match region.workspace_offset {
          Some(offset) => Json(offset)
          None => Json::null()
        },
        "bytes": Json(region.byte_count),
        "alignment": Json(region.alignment),
      }),
    ),
    "workspace_ownership": "one borrowed caller-owned arena, reusable serially across full-attention layers; persistent KV planes remain outside",
    "coefficient_storage_bytes": Json(
      operators.query_rotary().coefficient_bytes(),
    ),
    "rotary_geometry": Json(operators.query_rotary().geometry()),
    "axis_for_pair": Json(operators.query_rotary().axes()),
    "rotary_layout": "split halves of first 64 channels; preserve remaining 192; BF16 coefficients/products/sum",
    "output_gate_layout": Json(operators.full_output_gate().gate_layout()),
    "compiler_profile": "sm121 unqualified software profile; 256 threads, 49152 shared bytes, one pipeline/workgroup target, no async copy",
    "paged_abi_shape": Json(attention.abi().shape()),
    "paged_abi_layout": Json(attention.abi().layout()),
    "paged_abi_limits": Json(attention.abi().limits()),
    "cache_bytes_per_full_layer": Json(attention.cache_bytes()),
    "kernels": Json(kernels),
    "position_contract": "row-major T/H/W rotary coordinates are distinct from linear causal/cache token positions",
    "cache_contract": "publish rotated K and projected V using the existing paged writer before read-only causal attention in the same ordered queue; caller supplies valid counts/CSR/pages",
    "numerical_contract": "existing online-softmax FP32 attention over BF16 Q/K/V; not bitwise parity with eager BF16 score/probability materialization",
    "ingress_and_device_frame_implemented": true,
    "native_frame_executed": false,
    "cuda_compiled": false,
    "device_qualified": false,
  }
}
