///|
async fn load_attention_offline_input(
  input : (String, String, String)?,
  toolchain : String,
) -> (Bytes, String, String)? raise Qwen3Bf16CandidateExportCliError {
  match input {
    None => None
    Some((path, digest, device)) => {
      let bytes = @fs.read_file(path).binary() catch {
        _ => raise InvalidArguments
      }
      guard @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) == digest else {
        raise InvalidArguments
      }
      Some((bytes, device, toolchain))
    }
  }
}
