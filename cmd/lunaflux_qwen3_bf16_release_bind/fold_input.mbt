///|
/// Offline AOT input only: consume the same immutable observations and compiler
/// envelope as the exporter before regenerating the compiled-set contracts.
async fn load_release_fold_records(
  arguments : Qwen3ReleaseBindArguments,
) -> ReadOnlyArray[@luna_projection_tuning.ProjectionFoldRecord] raise Qwen3Bf16ReleaseBindError {
  match arguments.projection_folds {
    None => []
    Some((path, _, device)) => {
      let bytes = @fs.read_file(path).binary() catch {
        _ => raise InvalidArguments
      }
      @luna_projection_tuning.parse_projection_fold_records(
        bytes,
        scope=device + "\t" + arguments.toolchain_sha256,
      ) catch {
        _ => raise InvalidArguments
      }
    }
  }
}
