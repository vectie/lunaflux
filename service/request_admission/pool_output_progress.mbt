///|
fn LunaRequestPreparationPool::progress_output_setup(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let work = match lane.output_work {
    Some(value) => value
    None => abort("Luna output-work invariant")
  }
  let progress = work.progress() catch { _ => raise Invalid(OutputContract) }
  match progress {
    LunaIncrementalOutputPending => ()
    LunaIncrementalOutputReady => lane.phase = LUNA_PREPARATION_ASSEMBLE
  }
}
