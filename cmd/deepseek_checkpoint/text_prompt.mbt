///|
/// Literal installed-tokenizer input. The caller renders any chat template;
/// this frontend neither invents one nor reads checkpoint weight shards.
async fn prepare_text_frames(
  root : @approved_fs.ApprovedRoot,
  model_root : String,
  rows : Int,
  history : Int,
  tokenizer_label : String,
  input_path : String,
  output : String,
) -> Unit {
  let spec = inspect_config(root)
  guard rows > 0 &&
    rows <= history &&
    history > 1 &&
    history <= spec.max_position_embeddings() else {
    fail("invalid text-frame envelope")
  }
  let input = @text_input.load_checkpoint_text(
    root,
    model_root,
    "tokenizer.json",
    tokenizer_label,
    input_path,
    vocabulary_size=spec.vocabulary_size(),
    maximum_tokens=history - 1,
  )
  let ids = input.token_ids()
  let (limits, inference) = prompt_limits(rows, history, spec.vocabulary_size())
  let frames = @prompt.encode_prompt(
    ids,
    limits,
    inference,
    maximum_rows=rows,
    maximum_context=history,
    model_generation=1UL,
  )
  @fs.mkdir(output)
  for index, bytes in frames {
    @fs.write_file(output + "/plan-\{index}.bin", bytes, create_mode=CreateNew)
  }
  @fs.write_file(
    output + "/tokens.txt",
    ids.to_owned().map(id => id.to_string()).join(",") + "\n",
    create_mode=CreateNew,
  )
  @fs.write_file(
    output + "/decoded.txt",
    input.decoded_bytes(),
    create_mode=CreateNew,
  )
  println(
    "prompt_frames=\{frames.length()} input_tokens=\{ids.length()} tokenizer_authentication=false label_kind=caller_supplied",
  )
}
