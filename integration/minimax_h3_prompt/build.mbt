///|
/// H3 deliberately does not apply the Qwen chat template: no system message,
/// assistant prefix, BOS or EOS. Each label and the verbatim prompt are encoded
/// independently, exactly like the reference presentation accumulator.
pub fn build(
  tokenizer : @tokenizer.TokenizerSpec,
  config : @text_encoder.EncoderConfig,
  prompt~ : String,
  presentation : Presentation,
  max_tokens~ : Int,
) -> Prompt raise {
  guard prompt != "" &&
    max_tokens > 0 &&
    max_tokens <= 262144 &&
    max_tokens <= config.max_positions else {
    raise InvalidPrompt
  }
  for
    (id, piece) in [
      (config.vision_start_token, "<|vision_start|>"),
      (config.vision_end_token, "<|vision_end|>"),
      (config.image_token, "<|image_pad|>"),
      (config.video_token, "<|video_pad|>"),
    ] {
    let result = tokenizer.encode_text(
      piece,
      special_policy=Allow,
      output_limit=1,
      truncation=RejectOverflow,
    )
    guard result.token_ids == [id] else { raise InvalidTokenizer }
  }
  let ids : Array[Int] = []
  let tags : Array[Int] = []
  let grids : Array[@text_encoder.VisualGrid] = []
  fn text(value : String) -> Unit raise {
    let encoded = tokenizer.encode_text(
      value,
      special_policy=Allow,
      output_limit=max_tokens,
      truncation=RejectOverflow,
    )
    guard encoded.token_ids.length() <= max_tokens - ids.length() else {
      raise TokenBudgetExceeded
    }
    for id in encoded.token_ids {
      ids.push(id)
      tags.push(1)
    }
  }
  fn vision(grid : @text_encoder.VisualGrid, per_frame : Bool) -> Unit raise {
    let count = (if per_frame { 1 } else { grid.time }) *
      (grid.height / 2) *
      (grid.width / 2)
    guard count > 0 && count <= max_tokens - ids.length() - 2 else {
      raise TokenBudgetExceeded
    }
    ids.push(config.vision_start_token)
    tags.push(0)
    for _ in 0..<count {
      ids.push(
        if grid.kind == Image {
          config.image_token
        } else {
          config.video_token
        },
      )
      tags.push(0)
    }
    ids.push(config.vision_end_token)
    tags.push(0)
  }
  fn image(grid : @text_encoder.VisualGrid, ordinal : Int) -> Unit raise {
    guard grid.kind == Image && grid.time == 1 else { raise InvalidMedia }
    text("<Picture \{ordinal}>: ")
    vision(grid, false)
    grids.push(grid)
  }
  match presentation {
    Text => ()
    Keyframes(frames) => {
      let signature = frames.map(pair => pair.0)
      guard signature[:] == [0] ||
        signature[:] == [-1] ||
        signature[:] == [0, -1] else {
        raise InvalidMedia
      }
      for i, (_, grid) in frames {
        image(grid, i + 1)
      }
    }
    References(references) => {
      guard !references.is_empty() else { raise InvalidMedia }
      for item in references; images = 0, audios = 0, videos = 0 {
        match item {
          Image(grid) => {
            image(grid, images + 1)
            continue images + 1, audios, videos
          }
          Audio => {
            text("<Audio \{audios+1}>: ")
            continue images, audios + 1, videos
          }
          Video(grid, timestamps) => {
            guard grid.kind == Video && timestamps.length() == grid.time else {
              raise InvalidMedia
            }
            text("<Video \{videos+1}>: ")
            for ts in timestamps {
              text(timestamp(ts))
              vision(grid, true)
            }
            grids.push(grid)
            continue images, audios, videos + 1
          }
        }
      } nobreak {
        ()
      }
    }
  }
  text(prompt)
  let placement = if grids.is_empty() {
    let _ = config.plan(ids)
    None
  } else {
    Some(config.placement_from_grids(ids, grids))
  }
  {
    token_ids: ReadOnlyArray::from_array(ids),
    token_tags: ReadOnlyArray::from_array(tags),
    grids: ReadOnlyArray::from_array(grids),
    placement,
  }
}
