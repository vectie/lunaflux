///|
fn Builder::early_merger(
  self : Self,
  raw : BlockEntries,
  rows : Int32,
) -> Unit raise {
  let prefix = "model.vision_tower.vit_merger"
  let gather = self.add(
    "vision_early_gather",
    ((rows * 1152 + 255) / 256, 1, 256),
    4,
    @segmented.gather_source(
      maximum_rows=rows,
      width=1152,
      symbol="vision_early_gather",
    ),
  )
  let attention = self.attention("vision_window_attention", rows, 4)
  let norm = self.norm("vision_early_flat_norm", rows / 4, 4608)
  let up = self.linear("vision_early_up", rows / 4, 4608, 17216, 1)
  let down = self.linear("vision_early_down", rows / 4, 17216, 1152, 0)
  let residual = self.add(
    "vision_early_mean_residual",
    ((rows / 4 * 1152 + 255) / 256, 1, 256),
    4,
    @segmented.merge_residual_source(
      maximum_rows=rows,
      width=1152,
      symbol="vision_early_mean_residual",
    ),
  )
  self.norm_stage(
    raw.norm,
    RawCounts,
    State,
    prefix + ".layer_norm1",
    Normalized,
  )
  self.stage(gather, [
    Buffer(RawCounts),
    Buffer(WindowIndices),
    Buffer(Normalized),
    Buffer(Attention),
  ])
  for (name, output) in [("q_proj", Query), ("k_proj", Key), ("v_proj", Value)] {
    self.linear_stage(
      raw.square,
      RawCounts,
      Attention,
      prefix + ".self_attn." + name,
      output,
    )
  }
  self.attention_stage(attention, RawCounts, WindowStarts, WindowEnds)
  self.linear_stage(
    raw.square,
    RawCounts,
    Attention,
    prefix + ".self_attn.out_proj",
    Branch,
  )
  self.stage(gather, [
    Buffer(RawCounts),
    Buffer(WindowIndices),
    Buffer(State),
    Buffer(Normalized),
  ])
  self.stage(raw.residual, [
    Buffer(RawCounts),
    Buffer(Normalized),
    Buffer(Branch),
    Buffer(Attention),
  ])
  self.norm_stage(
    norm,
    EncodedCounts,
    Attention,
    prefix + ".pre_norm",
    Normalized,
  )
  self.linear_stage(
    up,
    EncodedCounts,
    Normalized,
    prefix + ".linear_1",
    Intermediate,
  )
  self.linear_stage(
    down,
    EncodedCounts,
    Intermediate,
    prefix + ".linear_2",
    Branch,
  )
  self.stage(residual, [
    Buffer(RawCounts),
    Buffer(Attention),
    Buffer(Branch),
    Buffer(State),
  ])
}

///|
pub fn Program::new(
  spec : @model.Spec,
  maximum_patches~ : Int32,
  maximum_crop_patches~ : Int32,
) -> Self raise {
  guard maximum_patches > 0 &&
    maximum_patches % 16 == 0 &&
    maximum_crop_patches > 0 &&
    maximum_crop_patches % 16 == 0 &&
    maximum_crop_patches <= maximum_patches &&
    maximum_patches <= 2147483647 / 17216 else {
    fail("invalid vision AOT row capacities")
  }
  ignore(@vision.attention_plan(spec, maximum_patches~, maximum_crop_patches~))
  let early = spec.downsample() is SixteenX
  let encoded = if early { maximum_patches / 4 } else { maximum_patches }
  let output = encoded / 4
  let builder = Builder::new()
  let raw = builder.block_entries(
    "vision_raw", maximum_patches, maximum_crop_patches,
  )
  let post = if early {
    builder.block_entries("vision_encoded", encoded, maximum_crop_patches / 4)
  } else {
    raw
  }
  let patches = builder.add(
    "vision_patch_rows",
    ((maximum_patches * 588 + 255) / 256, 1, 256),
    3,
    @segmented.patch_rows_source(
      maximum_rows=maximum_patches,
      channels=3,
      patch_size=14,
      symbol="vision_patch_rows",
    ),
  )
  let embed = builder.linear(
    "vision_patch_embed", maximum_patches, 588, 1152, 0,
  )
  let position = builder.add(
    "vision_position",
    ((maximum_patches * 1152 + 255) / 256, 1, 256),
    4,
    @segmented.position_add_source(
      maximum_rows=maximum_patches,
      width=1152,
      symbol="vision_position",
    ),
  )
  builder.stage(patches, [Buffer(RawCounts), Buffer(Pixels), Buffer(Normalized)])
  builder.linear_stage(
    embed,
    RawCounts,
    Normalized,
    "model.vision_tower.embeddings.patch_embedding",
    State,
  )
  builder.stage(position, [
    Buffer(RawCounts),
    Buffer(PositionIds),
    Weight("model.vision_tower.embeddings.position_embedding.weight"),
    Buffer(State),
  ])
  for layer in 0..<spec.vision().layers {
    let after = early && layer > spec.vision().compression_after_layer
    builder.block(
      if after {
        post
      } else {
        raw
      },
      if after {
        EncodedCounts
      } else {
        RawCounts
      },
      if after {
        EncodedStarts
      } else {
        RawStarts
      },
      if after {
        EncodedEnds
      } else {
        RawEnds
      },
      layer,
    )
    if early && layer == spec.vision().compression_after_layer {
      builder.early_merger(raw, maximum_patches)
    }
  }
  let count = if early { EncodedCounts } else { RawCounts }
  builder.norm_stage(
    post.norm,
    count,
    State,
    "model.vision_tower.post_layernorm",
    State,
  )
  let gather = builder.add(
    "vision_final_gather",
    ((encoded * 1152 + 255) / 256, 1, 256),
    4,
    @segmented.gather_source(
      maximum_rows=encoded,
      width=1152,
      symbol="vision_final_gather",
    ),
  )
  builder.stage(gather, [
    Buffer(count),
    Buffer(MergerIndices),
    Buffer(State),
    Buffer(Attention),
  ])
  let norm = builder.norm("vision_final_norm", output, 4608)
  let up = builder.linear("vision_final_up", output, 4608, 4608, 2)
  let down = builder.linear("vision_final_down", output, 4608, 2048, 0)
  let prefix = "model.merger.mlp.0"
  builder.norm_stage(
    norm,
    OutputCounts,
    Attention,
    prefix + ".pre_norm",
    Normalized,
  )
  builder.linear_stage(
    up,
    OutputCounts,
    Normalized,
    prefix + ".linear_1",
    Intermediate,
  )
  builder.linear_stage(
    down,
    OutputCounts,
    Intermediate,
    prefix + ".linear_2",
    Features,
  )
  let mut offset = 0L
  let regions = []
  for
    (plane, width) in [
      (State, 1152),
      (Normalized, 1152),
      (Query, 1152),
      (Key, 1152),
      (Value, 1152),
      (Attention, 1152),
      (Branch, 1152),
      (Intermediate, 4304),
    ] {
    let bytes = maximum_patches.to_int64() * width.to_int64() * 2L
    regions.push({ plane, offset, bytes, })
    offset += bytes
  }
  {
    source: @utf8.encode(builder.source.to_string()),
    entries: ReadOnlyArray::from_array(builder.entries),
    stages: ReadOnlyArray::from_array(builder.stages),
    regions: ReadOnlyArray::from_array(regions),
    workspace_bytes: offset,
    rows: (maximum_patches, encoded, output),
    crop_rows: maximum_crop_patches,
  }
}
