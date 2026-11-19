///|
/// Offline model/weight metadata join and initial operator source export.
/// Neither weights nor a device runtime are loaded by this command.
async fn main {
  let args = @env.args()
  guard args[1:] is [assets, headers, output] && !@fs.exists(output) else {
    fail(
      "usage: minicpm_v47_prepare ASSETS_DIRECTORY HEADERS_DIRECTORY NEW_OUTPUT_DIRECTORY",
    )
  }
  let model = @model.Spec::parse(
    @fs.read_file(assets + "/config.json").binary()[:],
  )
  guard @fs.read_file(assets + "/model.safetensors.index.json").json()
    is { "weight_map": Object(index), .. } &&
    @fs.read_file(assets + "/shards.json").json() is Array(tree) else {
    fail("missing model shard metadata")
  }
  let files : Map[String, Int64] = Map([])
  for entry in tree {
    if entry is { "path": String(name), "size": Number(size, ..), .. } &&
      name.has_suffix(".safetensors") {
      files[name] = size.to_int64()
    }
  }
  let names = files.keys().to_array()
  names.sort()
  let limits = @safetensors.HeaderLimits::new(
    max_header_bytes=1048576,
    max_tensor_count=4096,
    max_rank=4,
    max_name_length=256,
    max_payload_bytes=80000000000L,
    max_materialized_tensor_bytes=80000000000L,
  )
  let shards = []
  for name in names {
    let bytes = @fs.read_file(headers + "/" + name + ".header.json").binary()
    let prefix = @buffer.Buffer()
    prefix.write_uint64_le(bytes.length().to_uint64())
    prefix.write_bytes(bytes)
    shards.push(
      @safetensors_reader.read_header(
        prefix.to_bytes()[:],
        payload_bytes=files[name] - 8L - bytes.length().to_int64(),
        limits~,
      ),
    )
  }
  let bound = @weights.bind(model, shards[:])
  // Source index and parsed headers must name the same owning shard.
  guard index.length() == bound.tensor_count() else {
    fail("source index and tensor metadata counts differ")
  }
  for i in 0..<bound.tensor_count() {
    let tensor = bound.tensor(i).unwrap()
    guard index.get(tensor.name) == Some(Json(names[tensor.shard_index])) else {
      fail("source index assigns a tensor to a different shard")
    }
  }
  let geometry = model.decoder()
  let recurrent = geometry.recurrent_attention()
  let state_bytes = geometry.fp32_recurrent_matrix_bytes(1)
  let channels = 2 * recurrent.key_heads * recurrent.key_width +
    recurrent.value_heads * recurrent.value_width
  let operators = @operators.DecoderOperators::new(model, maximum_rows=32)
  let compilation = @bootstrap.compile_spark_attention(operators)
  let text_program = @bootstrap.text_program(operators, compilation)
  let decoder = text_program.decoder_program()
  let moe = decoder.moe_program()
  let convolution = operators.convolution()
  @fs.mkdir(output, recursive=true)
  let recurrent_report = prepare_recurrent_sources(
    decoder.recurrent_program(),
    output,
  )
  let normalizations : Array[(String, @operators.NormSite, String)] = [
    ("decoder_norm", Decoder, "decoder_envelope.cu"),
    ("query_norm", FullQuery, "attention_common.cu"),
    ("key_norm", FullKey, "attention_common.cu"),
    ("recurrent_norm", RecurrentOutput, "recurrent_branch.cu"),
  ]
  let normalization_reports : Array[Json] = []
  for (name, site, source) in normalizations {
    let plan = operators.normalization(site)
    normalization_reports.push({
      "site": Json(name),
      "source": Json(source),
      "geometry": Json(plan.geometry()),
      "input_layout_bf16_elements": Json(plan.input_layout()),
      "output_bytes": Json(plan.output_bytes()),
      "output_layout": "packed",
      "weight_law": Json(
        if site is RecurrentOutput {
          "direct gamma; BF16 norm and weight rounds; SiLU gate"
        } else {
          "FP32 one-plus-gamma; final BF16 round"
        },
      ),
    })
  }
  let router_stage = moe.stages()[1]
  let expert_report = prepare_expert_sources(
    operators,
    moe,
    bound,
    geometry.layer_count(),
    output,
  )
  let attention_report = prepare_attention_sources(
    operators,
    compilation,
    decoder.attention_program(),
    output,
  )
  let decoder_report = prepare_decoder_sources(decoder, output)
  let text_report = prepare_text_sources(text_program, output)
  let vision_report = prepare_vision_sources(model, bound, output)
  let multimodal_report = prepare_multimodal_sources(text_program, output)
  let report : Json = {
    "format": "lunaflux.minicpm-v47-preparation.v1",
    "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
    "revision": "98e79ba972b586ac6146d3505771ce1467e199ba",
    "scope": "inspected headers and uncompiled operator sources",
    "tensor_count": Json(bound.tensor_count()),
    "weight_bytes": Json(bound.total_bytes()),
    "source_shards": Json(names),
    "context_capacity": 4096,
    "bf16_kv_bytes": Json(geometry.bf16_kv_bytes(1, 4096)),
    "fp32_recurrent_matrix_bytes": Json(state_bytes),
    "bf16_convolution_state_bytes": Json(
      convolution.state_bytes() * geometry.recurrent_layer_count().to_int64(),
    ),
    "recurrent_branch": recurrent_report,
    "normalizations": Json(normalization_reports),
    "router": {
      "source": "moe.cu",
      "symbol": Json(router_stage.symbol()),
      "launch": Json(router_stage.launch()),
      "parameters": Json(router_stage.arguments().length()),
      "geometry": Json(operators.routing().geometry()),
      "input": "BF16 full router logits",
      "weights": "BF16-rounded selected-renormalized full FP32 softmax, exactly widened into F32 storage",
      "ties": "lower expert ID; upstream topk tie order is unspecified",
    },
    "arithmetic_qualification": "ordered reduction implementations; upstream GPU numerical comparisons pending",
    "experts": expert_report,
    "full_attention": attention_report,
    "decoder_stack": decoder_report,
    "text_forward": text_report,
    "vision_encoder": vision_report,
    "multimodal_prefill": multimodal_report,
    "qkv_row_stride_bf16_elements": Json(channels),
    "query_byte_offset": "0",
    "key_byte_offset": Json(
      (recurrent.key_heads * recurrent.key_width * 2).to_int64(),
    ),
    "value_byte_offset": Json(
      (recurrent.key_heads * recurrent.key_width * 4).to_int64(),
    ),
    "decoder_rows_per_frame": 32,
    "sequence_slots": 1,
    "complete_model_execution": false,
    "cuda_compiled": false,
    "device_qualified": false,
  }
  @fs.write_file(
    output + "/preparation.json",
    @utf8.encode(report.stringify(indent=2)),
    create_mode=CreateNew,
  )
  println(report.stringify())
}
