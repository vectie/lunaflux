///|
/// Decoder token/position carrier. Its historical name also covers explicit
/// multimodal placeholders; feature construction/insertion belongs outside.
pub struct PreparedTextPrompt {
  priv tokens : ReadOnlyArray[Int32]
  priv positions : @positions.Sequence
}

///|
pub fn PreparedTextPrompt::tokens(self : Self) -> ReadOnlyArray[Int32] {
  self.tokens
}

///|
pub fn PreparedTextPrompt::positions(self : Self) -> @positions.Sequence {
  self.positions
}

///|
/// Request-preparation bridge for the initial stateless planner profile. The
/// caller supplies its admitted context budget. No truncation and no visual
/// placeholder inference; multimodal processors build their own modality/grid
/// inputs for the same position owner.
pub fn prepare_text_prompt(
  model : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  system : String,
  user : String,
  maximum_tokens~ : Int32,
  system_present~ : Bool = true,
) -> PreparedTextPrompt raise {
  guard maximum_tokens > 0 && maximum_tokens <= model.decoder().context_length() else {
    fail("invalid prompt context budget")
  }
  let encoded = tokenizer.encode_text(
    render_text_prompt(system, user, system_present~),
    special_policy=Allow,
    output_limit=maximum_tokens,
    truncation=RejectOverflow,
  )
  prepare_token_prompt(model, encoded.token_ids, maximum_tokens~)
}

///|
/// Native token-ID text ingress. IDs already include the caller's template;
/// no BOS, visual embeddings or implicit truncation are introduced here.
pub fn prepare_token_prompt(
  model : @model.Spec,
  tokens : ArrayView[Int32],
  maximum_tokens~ : Int32,
) -> PreparedTextPrompt raise {
  prepare_positioned_prompt(
    model,
    tokens,
    Array::make(tokens.length(), @positions.Modality::Text),
    image_grids=[],
    video_grids=[],
    maximum_tokens~,
  )
}

///|
/// Explicit processor-owned modalities/grids; never guess image presence from
/// token IDs. No BOS, image pixels, truncation or implicit template is added.
pub fn prepare_positioned_prompt(
  model : @model.Spec,
  tokens : ArrayView[Int32],
  modalities : ArrayView[@positions.Modality],
  image_grids~ : ArrayView[@positions.Grid],
  video_grids~ : ArrayView[@positions.Grid],
  maximum_tokens~ : Int32,
) -> PreparedTextPrompt raise {
  guard maximum_tokens > 0 &&
    maximum_tokens <= model.decoder().context_length() &&
    !tokens.is_empty() &&
    tokens.length() <= maximum_tokens &&
    tokens.all(token => token >= 0 && token < model.decoder().vocabulary_size()) else {
    fail("invalid native text prompt")
  }
  let positions = @positions.Sequence::build(
    model,
    tokens,
    modalities,
    image_grids~,
    video_grids~,
  )
  { tokens: ReadOnlyArray::from_array(tokens), positions, }
}
