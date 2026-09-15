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
) -> String {
  let effects = plan.effect_plan(
    if asynchronous {
      OverlappedTransfer
    } else {
      SerialTransfer
    },
  )
  let (stages, width, extent) = effects.geometry()
  let asynchronous = effects.overlaps()
  // Keep the established two-slot realization byte-for-byte stable at its
  // scheduling seam. Wider rings are explicit offline alternatives.
  if stages == 2 {
    let first_phase = if phase_argument { ", false" } else { "" }
    let next_phase = if phase_argument { ", true" } else { "" }
    let copy = "if (inner + \{width} < \{extent}) stage_operands(inner + \{width}, stage ^ 1\{next_phase});"
    let wait = if asynchronous {
      "asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");"
    } else {
      ""
    }
    let prime = if asynchronous {
      "stage_operands(0, 0\{first_phase});\n      \{wait}"
    } else {
      "stage_operands(0, 0\{first_phase});"
    }
    return (
      $|      \{prime}
      $|      __syncthreads();
      $|      int stage = 0;
      $|      #pragma unroll 1
      $|      for (int inner = 0; inner < \{extent}; inner += \{width}) {
      $|        \{if asynchronous { copy } else { "" }}
      $|        \{body}
      $|        \{if asynchronous { wait } else { copy }}
      $|        __syncthreads();
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
  (
    $|      #pragma unroll
    $|      for (int prime = 0; prime < \{effects.prime_count()}; ++prime) stage_operands(prime * \{width}, prime\{phase});
    $|      \{prime_wait}
    $|      __syncthreads();
    $|      int stage = 0;
    $|      #pragma unroll 1
    $|      for (int inner = 0; inner < \{extent}; inner += \{width}) {
    $|        \{if asynchronous { prefetch } else { "" }}
    $|        \{body}
    $|        \{if asynchronous { wait } else { prefetch }}
    $|        __syncthreads();
    $|        stage = (stage + 1) % \{stages};
    $|      }
  )
}
