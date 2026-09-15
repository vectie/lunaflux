///|
/// Terminal realization of a finite immutable operand ring. Publication and
/// reuse are CTA-uniform; the ordered accumulator body is inserted unchanged.
/// A synchronous producer publishes after consumption, while an asynchronous
/// producer starts before it. Neither can overwrite the current live slot.
fn projection_fold_lifetime_source(
  plan : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  body : String,
  asynchronous~ : Bool,
  phase_argument? : Bool = false,
  completed_on_issue? : Bool = false,
) -> String {
  let effects = plan.effect_plan(
    if asynchronous && completed_on_issue {
      CompletedPrefetch
    } else if asynchronous {
      OverlappedTransfer
    } else {
      SerialTransfer
    },
  )
  let (stages, width, extent) = effects.geometry()
  let asynchronous = effects.has_pending_transfers()
  // Completion and issue order are independent: a synchronous prefetch may
  // retain its early issue position without empty async commit/wait groups.
  if stages == 2 {
    let first_phase = if phase_argument { ", false" } else { "" }
    let next_phase = if phase_argument { ", true" } else { "" }
    let copy = "if (inner + \{width} < \{extent}) stage_operands(inner + \{width}, stage ^ 1\{next_phase});"
    let wait = if asynchronous {
      "asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");"
    } else {
      ""
    }
    let prime = projection_fold_bootstrap_source(
      effects,
      "stage_operands(0, 0\{first_phase});",
      wait,
    )
    return (
      $|      \{prime}
      $|      int stage = 0;
      $|      #pragma unroll 1
      $|      for (int inner = 0; inner < \{extent}; inner += \{width}) {
      $|        \{projection_fold_effects_source(effects, body, copy, wait)}
      $|        stage ^= 1;
      $|      }
    )
  }
  let phase = if phase_argument { ", prime != 0" } else { "" }
  let next_phase = if phase_argument { ", true" } else { "" }
  let prefetch = "if (inner + \{effects.future_distance()} < \{extent}) stage_operands(inner + \{effects.future_distance()}, (stage + \{effects.prime_count()}) % \{stages}\{next_phase});"
  let prime_wait = if asynchronous {
    "asm volatile(\"cp.async.wait_group " +
    effects.pending_group_limit().to_string() +
    ";\" ::: \"memory\");"
  } else {
    ""
  }
  let wait = if asynchronous {
    "if (inner + " +
    effects.future_distance().to_string() +
    " < " +
    extent.to_string() +
    ") { " +
    prime_wait +
    " } else { asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\"); }"
  } else {
    ""
  }
  let issue = "#pragma unroll\n      for (int prime = 0; prime < \{effects.prime_count()}; ++prime) stage_operands(prime * \{width}, prime\{phase});"
  let prime = projection_fold_bootstrap_source(effects, issue, prime_wait)
  (
    $|      \{prime}
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{extent}; inner += \{width}) {
    $|        \{projection_fold_effects_source(effects, body, prefetch, wait)}
    $|        stage = (stage + 1) % \{stages};
    $|      }
  )
}

///|
fn projection_fold_bootstrap_source(
  effects : @luna_projection_tile_compiler.ProjectionFoldEffectPlan,
  issue : String,
  wait : String,
) -> String {
  let lines : Array[String] = []
  for action in effects.bootstrap() {
    match action {
      IssueInitialOperands => lines.push(issue)
      AwaitInitialOperands => lines.push(wait)
      PublishInitialOperands => {
        // Preserve the existing wider serial ring's empty wait line.
        if !effects.overlaps() && effects.prime_count() > 1 {
          lines.push("")
        }
        lines.push("__syncthreads();")
      }
    }
  }
  lines.join("\n      ")
}

///|
/// Instruction rendering only: semantic action order comes from the plan.
fn projection_fold_effects_source(
  effects : @luna_projection_tile_compiler.ProjectionFoldEffectPlan,
  body : String,
  transfer : String,
  wait : String,
) -> String {
  let lines : Array[String] = []
  for action in effects.iteration() {
    match action {
      IssueFutureOperand => lines.push(transfer)
      ConsumeCurrentOperand => {
        // Preserve the established source formatting when there is no
        // pre-consume transfer; this empty line has no execution effect.
        if lines.is_empty() {
          lines.push("")
        }
        lines.push(body)
      }
      AwaitOperandTransfers => lines.push(wait)
      PublishOperandSlot => lines.push("__syncthreads();")
    }
  }
  lines.join("\n        ")
}
