///|
// Hugging Face BPE serialization may omit this field; the algorithm's default
// is false. Absence is not permission to coerce null, strings or numbers.
fn optional_ignore_merges(
  model : Map[String, Json],
) -> Bool raise TokenizerJsonError {
  match model.get("ignore_merges") {
    None | Some(False) => false
    Some(True) => true
    _ => raise InvalidField(component=Model, field=IgnoreMerges, index=0)
  }
}
