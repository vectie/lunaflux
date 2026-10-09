///|
async fn main {
  if @env.args() is [_, "config", model_root] {
    let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
    defer root.close()
    let spec = inspect_config(root)
    println(
      "GLM-5.3 Flash config: layers=\{spec.layer_count()} hidden=\{spec.hidden_size()} vocabulary=\{spec.vocabulary_size()}",
    )
    return
  }
  guard @env.args()
    is [
      _,
      mode,
      model_root,
      inventory_name,
      rows,
      history,
      budget0,
      budget1,
      reserve,
      .. tail,
    ] else {
    fail(
      "MODE MODEL_ROOT SHARD_SHA256 ROWS HISTORY BUDGET0 BUDGET1 RESERVE [MODE_ARGS]",
    )
  }
  // Reject mode/arguments before opening or hashing checkpoint payloads.
  let mode_valid = match (mode, tail.length()) {
    ("export", 1) | ("egress", 4) | ("ingress", 7) | ("generate", 9) => true
    _ => false
  }
  guard mode_valid else {
    fail(
      "export OUTDIR | egress AOT_ROOT AOT_FILE CONTROL_LISTEN ACTIVATION_LISTEN | ingress AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT FRAME_COUNT OUTDIR | generate AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT PREFILL_COUNT MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE OUTDIR",
    )
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let startup = inspect_startup(
    root,
    inventory_name,
    @string.parse_int(rows),
    @string.parse_int(history),
    [@string.parse_int64(budget0), @string.parse_int64(budget1)],
    @string.parse_int64(reserve),
  )
  for ordinal, range in startup.placement.stages() {
    let (first, end) = range.layer_range()
    println(
      "stage=\{ordinal} layers=[\{first},\{end}) planned_bytes=\{startup.placement.stage_bytes()[ordinal]} ceiling=\{startup.budgets[ordinal]}",
    )
  }
  match mode {
    "export" => {
      @fs.mkdir(tail[0])
      for ordinal in 0..<2 {
        let stage = stage_plan(startup, ordinal)
        defer stage.close()
        @fs.write_file(
          tail[0] + "/stage-\{ordinal}.cu",
          stage.source_bytes(),
          create_mode=CreateNew,
        )
      }
      @fs.write_file(
        tail[0] + "/placement.txt",
        placement_receipt(startup),
        create_mode=CreateNew,
      )
    }
    "egress" => run_egress(startup, root, tail)
    "ingress" => run_ingress(startup, root, tail, generate=false)
    "generate" => run_ingress(startup, root, tail, generate=true)
    _ => fail("invalid mode")
  }
}

///|
fn placement_receipt(startup : Startup) -> String {
  let builder = StringBuilder()
  builder <+
    "glm-checkpoint-placement-v1\nrows=\{startup.rows}\nhistory=\{startup.history}\nreserve=\{startup.reserve}\n"
  for ordinal, range in startup.placement.stages() {
    let (first, end) = range.layer_range()
    builder <+
      "stage=\{ordinal} first=\{first} end=\{end} planned=\{startup.placement.stage_bytes()[ordinal]} ceiling=\{startup.budgets[ordinal]}\n"
  }
  builder.to_string()
}
