///|
async fn load_attention_offline_input(
  input : (String, String, String)?,
  toolchain : String,
) -> (Bytes, String, String)? raise Qwen3Bf16CandidateExportCliError {
  match input {
    None => None
    Some((path, _, device)) => {
      let bytes = @fs.read_file(path).binary() catch {
        _ => raise InvalidArguments
      }
      // Supplied identity labels select measurements; semantic parsing in the
      // tuning layer remains responsible for compatible shapes and resources.
      Some((bytes, device, toolchain))
    }
  }
}
