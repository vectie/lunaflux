///|
/// Immutable startup products, not native resource owners. Deployment stages
/// the pinned config/tokenizer/index/shards in the supplied read-only root.
pub(all) struct PreparedModel {
  model : @model.Spec
  tokenizer : @tokenizer.TokenizerSpec
  program : @text.Program
  weights : @weights.Plan
  inspected_tensors : Int32
  inspected_shards : Int32
}

///|
pub fn prepare(root : @approved_fs.ApprovedRoot) -> PreparedModel raise {
  prepare_checkpoint(root).0
}

///|
fn prepare_checkpoint(
  root : @approved_fs.ApprovedRoot,
) -> (PreparedModel, @checkpoint.PackedCheckpoint) raise {
  let model = @model.Spec::parse(
    read_asset(root, "config.json", maximum_bytes=1048576L)[:],
  )
  let tokenizer = @prompt.prepare_tokenizer(
    model,
    read_asset(root, "tokenizer.json", maximum_bytes=33554432L)[:],
    @tokenizer.TokenizerDigest::parse("0".repeat(64)),
    {
      max_input_bytes: 32768,
      max_output_tokens: 4096,
      max_decoded_bytes: 16777216,
      max_vocab_entries: 250000,
      max_merge_rules: 500000,
      max_token_bytes: 2048,
      max_special_tokens: 128,
    },
  )
  let inspection = inspect_checkpoint(
    root,
    @json_guard.parse_bounded(
      @utf8.decode(
        read_asset(root, "model.safetensors.index.json", maximum_bytes=4194304L),
      ),
      max_depth=32,
    ),
  )
  let operators = @operators.DecoderOperators::new(model, maximum_rows=32)
  let attention = compile_spark_attention(operators)
  let checkpoint = @checkpoint.PackedCheckpoint::new(inspection)
  let prepared : PreparedModel = {
    model,
    tokenizer,
    program: text_program(operators, attention),
    weights: @weights.Plan::new(operators, checkpoint),
    inspected_tensors: inspection.total_tensor_count(),
    inspected_shards: inspection.shard_count(),
  }
  (prepared, checkpoint)
}

///|
/// Both model branches from one config/tokenizer/header inspection. This does
/// not allocate GPU memory or read full tensor payloads.
pub(all) struct PreparedImageModel {
  text : PreparedModel
  vision : @vision.Program
  vision_weights : @vision.WeightPlan
}

///|
pub fn prepare_images(
  root : @approved_fs.ApprovedRoot,
  maximum_patches~ : Int32,
  maximum_crop_patches~ : Int32,
) -> PreparedImageModel raise {
  let (text, checkpoint) = prepare_checkpoint(root)
  let vision = @vision.Program::new(
    text.model,
    maximum_patches~,
    maximum_crop_patches~,
  )
  {
    text,
    vision,
    vision_weights: @vision.WeightPlan::new(text.model, vision, checkpoint),
  }
}
