///|
fn operator_command_header(command : StringView) -> String {
  match command {
    "run" => "LunaFlux legacy compatibility model run preflight"
    "doctor" => "LunaFlux legacy compatibility physical diagnostic"
    "plan" => "LunaFlux legacy compatibility model startup plan"
    "bench" => "LunaFlux legacy compatibility model benchmark preflight"
    "inspect-kernels" =>
      "LunaFlux legacy compatibility kernel inspection preflight"
    _ => abort("LunaFlux operator command invariant")
  }
}

///|
fn operator_command_terminal_line(command : StringView) -> String {
  match command {
    "run" => "service started: false"
    "doctor" => "diagnostic report available: false"
    "plan" => "resolved plan available: false"
    "bench" => "benchmark started: false"
    "inspect-kernels" => "kernel admission available: false"
    _ => abort("LunaFlux operator command invariant")
  }
}

///|
/// Runs the explicitly named legacy MODEL-root and CUDA preflight. It cannot publish readiness,
/// a resolved model plan, benchmark evidence, or kernel metadata; callers that
/// possess independently approved model/kernel roots and a pinned descriptor
/// use `run_pinned_operator_command` below.
fn run_operator_command(command : String, model_root : String) -> Unit raise {
  let outcome = @model_startup.preflight(model_root)
  println(operator_command_header(command))
  println(@model_startup.outcome_line(outcome))
  println(operator_command_terminal_line(command))
  println("readiness: false")
}

///|
/// Maps only explicit legacy command names to the older preflight vocabulary.
/// Bare canonical spellings are intentionally excluded so malformed pinned
/// operands cannot fall back to weaker root admission.
fn legacy_pinned_command(command : StringView) -> String? {
  match command {
    "legacy-run" => Some("run")
    "legacy-doctor" => Some("doctor")
    "legacy-plan" => Some("plan")
    "legacy-bench" => Some("bench")
    "legacy-inspect-kernels" => Some("inspect-kernels")
    _ => None
  }
}

///|
fn pinned_terminal_line(command : StringView, admitted : Bool) -> String {
  if admitted {
    match command {
      "plan" => "resolved plan available: true"
      "doctor" => "diagnostic report available: true"
      "inspect-kernels" => "kernel admission available: true"
      _ => operator_command_terminal_line(command)
    }
  } else {
    operator_command_terminal_line(command)
  }
}

///|
fn runtime_report_rejection_line(
  error : @runtime_report.RuntimeReportError,
) -> String {
  match error {
    InvalidAdmission => "runtime report rejected: invalid admitted scalars"
    InconsistentAdmission =>
      "runtime report rejected: inconsistent execution ceilings"
    DeviceBytesOverflow => "runtime report rejected: device byte total overflow"
  }
}

///|
fn print_pinned_admission(
  command : StringView,
  admission : @descriptor_file.RuntimeDescriptorAdmission,
) -> Bool {
  let report = @runtime_report.build(admission) catch {
    error => {
      println(runtime_report_rejection_line(error))
      return false
    }
  }
  let identity = admission.model_identity()
  let target = admission.target()
  println(
    "runtime descriptor sha256: \{admission.descriptor_digest().as_hex()}",
  )
  println("model content sha256: \{identity.content().as_hex()}")
  println("model plan sha256: \{identity.plan().as_hex()}")
  println("assigned device ordinal: \{admission.device_ordinal()}")
  println(
    "admitted target: sm_\{target.compute_major()}\{target.compute_minor()}",
  )
  println("admitted bootstrap sha256: \{admission.bootstrap_digest().as_hex()}")
  println(
    "bootstrap source sha256: \{admission.bootstrap_source().digest().as_hex()}",
  )
  if command == "inspect-kernels" {
    println(@kernel_inspection.render_paged(admission.execution()))
  }
  println(@runtime_report.render(report))
  true
}

///|
fn run_pinned_operator_command(
  command : String,
  model_root : String,
  kernel_root : String,
  descriptor_relative : String,
  descriptor_sha256 : String,
) -> Unit raise {
  let preflight = @model_startup.preflight_pinned(
    model_root, kernel_root, descriptor_relative, descriptor_sha256,
  )
  println(operator_command_header(command))
  println(@model_startup.outcome_line(preflight.outcome()))
  match preflight.admission() {
    Some(admission) => {
      let reported = print_pinned_admission(command, admission)
      println(pinned_terminal_line(command, reported))
    }
    None => println(pinned_terminal_line(command, false))
  }
  println("readiness: false")
}
