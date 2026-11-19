///|
pub(all) enum Content {
  Text(String)
  Image(@processor.Plan)
}

///|
pub struct ImagePrompt {
  priv prompt : @prompt.PreparedTextPrompt
  priv feature_map : ReadOnlyArray[Int32]
  priv plans : ReadOnlyArray[@processor.Plan]
  priv features : Int32
}

///|
pub fn ImagePrompt::prompt(self : Self) -> @prompt.PreparedTextPrompt {
  self.prompt
}

///|
pub fn ImagePrompt::feature_map(self : Self) -> ReadOnlyArray[Int32] {
  self.feature_map
}

///|
pub fn ImagePrompt::image_plans(self : Self) -> ReadOnlyArray[@processor.Plan] {
  self.plans
}

///|
pub fn ImagePrompt::feature_rows(self : Self) -> Int32 {
  self.features
}

///|
/// Stateless pinned system/user profile, content parts joined with newlines,
/// local image IDs starting at zero, no tools/thinking output. Tokenize once
/// so text BPE boundaries match the fully rendered prompt.
pub fn prepare_image_prompt(
  spec : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  system : String,
  content : ArrayView[Content],
  maximum_tokens~ : Int32,
  system_present~ : Bool = true,
) -> ImagePrompt raise {
  guard maximum_tokens > 0 && maximum_tokens <= spec.decoder().context_length() else {
    fail("invalid image prompt context budget")
  }
  let body = []
  let plans = []
  let mut features = 0
  for part in content {
    match part {
      Text(text) => body.push(text)
      Image(plan) => {
        guard plan.visual_tokens() <= maximum_tokens - features else {
          fail("image embeddings exceed prompt context budget")
        }
        features += plan.visual_tokens()
        body.push(plan.placeholder_text(image_id=plans.length()))
        plans.push(plan)
      }
    }
  }
  guard !plans.is_empty() else {
    fail("image prompt requires explicit image content")
  }
  let rendered = @prompt.render_text_prompt(
    system,
    body.join("\n"),
    system_present~,
  )
  let tokens = tokenizer.encode_text(
      rendered,
      special_policy=Allow,
      output_limit=maximum_tokens,
      truncation=RejectOverflow,
    ).token_ids
  prepare_image_tokens(spec, tokens[:], plans[:], maximum_tokens~)
}

///|
/// Already-templated token ingress with explicit image plans in occurrence
/// order. The same binding law is shared with the text/content-list path.
pub fn prepare_image_tokens(
  spec : @model.Spec,
  tokens : ArrayView[Int32],
  image_plans : ArrayView[@processor.Plan],
  maximum_tokens~ : Int32,
) -> ImagePrompt raise {
  guard !image_plans.is_empty() &&
    maximum_tokens > 0 &&
    maximum_tokens <= spec.decoder().context_length() else {
    fail("invalid native image prompt")
  }
  let grids = []
  let mut features = 0
  for plan in image_plans {
    guard plan.visual_tokens() <= maximum_tokens - features else {
      fail("image features exceed context budget")
    }
    features += plan.visual_tokens()
    for grid in plan.target_sizes() {
      grids.push(grid)
    }
  }
  let modalities = Array::make(tokens.length(), @positions.Modality::Text)
  let feature_map = Array::make(tokens.length(), -1)
  let mut index = 0
  for row in 0..<tokens.length() {
    if tokens[row] == spec.markers().image {
      modalities[row] = @positions.Modality::Image
      feature_map[row] = index
      index += 1
    }
  }
  // Additional literal image placeholders in text cannot consume image rows.
  guard index == features else {
    fail("image placeholder count differs from supplied image plans")
  }
  let prompt = @prompt.prepare_positioned_prompt(
    spec,
    tokens[:],
    modalities[:],
    image_grids=grids[:],
    video_grids=[],
    maximum_tokens~,
  )
  {
    prompt,
    feature_map: ReadOnlyArray::from_array(feature_map),
    plans: ReadOnlyArray::from_array(image_plans.to_owned()),
    features,
  }
}
