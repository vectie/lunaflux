///|
fn require_model_root_binding(
  root : @approved_fs.ApprovedRoot,
  label : StringView,
) -> Unit raise RootBoundWorkerProcessError {
  try root.require_absolute_identity(label) catch {
    error => raise RootBoundFailure(ModelRootBinding(error))
  } noraise {
    _ => ()
  }
}

///|
fn require_kernel_root_binding(
  root : @approved_fs.ApprovedRoot,
  label : StringView,
) -> Unit raise RootBoundWorkerProcessError {
  try root.require_absolute_identity(label) catch {
    error => raise RootBoundFailure(KernelRootBinding(error))
  } noraise {
    _ => ()
  }
}
