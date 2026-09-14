///|
async fn write_new_candidate_pair(
  root : String,
  source : Bytes,
  recipe : Bytes,
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  @fs.mkdir(root, permission=0o700) catch {
    _ => raise Publication
  }
  @fs.write_file(
    root + "/kernel.cu",
    source,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
  @fs.write_file(
    root + "/kernel.recipe",
    recipe,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
}
