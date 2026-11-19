# MiniCPM-V 4.7 resident text runtime

`Execution` owns the seven loaded AOT modules, stream, text storage/frame and
two request queues. Immutable weights and the device context are borrowed.
`Resident` composes that execution owner with the existing bounded streaming
weight loader. Neither owner imports HTTP, training, actor or E1 control code.

`ModuleSet` owns the shared seven-module load/bind/close boundary used by both
this text runtime and the separate image runtime. Its bindings are borrowed;
close every borrowing queue/frame before closing the set.

```mbt check
///|
test "unprepared modules expose no bindings and close idempotently" {
  let modules = @minicpm_v47_runtime.ModuleSet::new()
  assert_true((try? modules.bindings()) is Err(_))
  modules.close()
  modules.close()
  assert_true((try? modules.bindings()) is Err(_))
}
```

The caller builds one text program and matching weight plan, creates a resident
owner, and retains it even when preparation fails. Admit `Resident::budget()`
against the supplied allocation ceiling, read `ModuleImages` from the offline
build's CUBIN directory, then prepare once. Obtain the borrowed request runner
and use it repeatedly. Close the resident owner before the context. Partial
startup failures retain cleanup authority; close may be retried. There is no
runtime compiler, payload hash, duplicate expert bank or full host weight arena.

For a caller that already prepared the matching program, weights, modules and
prompt (all input handles remain borrowed):

```mbt check
///|
async fn _one_request(
  program : @text.Program,
  weights : @weights.Plan,
  context : @device.Context,
  model_root : @approved_fs.ApprovedRoot,
  images : @minicpm_v47_runtime.ModuleImages,
  scratch : FixedArray[Byte],
  limits : @streaming.StreamingSafetensorsLimits,
  maximum_device_bytes : Int64,
  prompt : @prompt.PreparedTextPrompt,
) -> Unit {
  @async.with_task_group() <| group => {
    let resident = @minicpm_v47_runtime.Resident::new(program, weights)
    group.add_defer(() => resident.close())
    resident.prepare(
      context,
      model_root,
      images,
      maximum_device_bytes~,
      scratch~,
      limits~,
    )
    let runner = resident.runner()
    runner.begin(
      prompt,
      maximum_new_tokens=8,
      stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
    )
    while runner.submit_next() {
      runner.wait()
    }
  }
}
```

A service retains that resident scope across requests and repeats only
`begin`/submit/completion/output handling. A caller event loop can use `poll`
instead of blocking `wait`; scheduling, deadlines and output publication stay
outside the resident resource owner.

Budgets cover explicit weight/storage/output payload allocations; module,
driver and queue overhead still requires device headroom. At the initial
32-row/4096-token profile they total 69,473,047,868 bytes, including the
69,319,779,584-byte text weights. Vision remains outside this text owner.

`ModuleImages::read` reads the seven named CUBINs once under a per-module size
ceiling. Deployment owns their immutable association with the exact program
and target. Reading a compiler receipt or loading modules does not establish
numerical compatibility. No filesystem or source/build work enters token steps.

Four focused native tests cover exact payload arithmetic, fresh owner lifecycle,
the seven-module reader and its failure cleanup, and fresh whole-resident binding
to all real header fixtures. The separate native ABI fixture prepares a complete
execution session over borrowed synthetic weights, exercises repeated requests
and closes it with work active. Admission and partial-startup cleanup also pass.
Its local logs are `target/minicpm-v47-runtime-abi.log` and
`target/minicpm-v47-runtime-asan.log`. The ASan smoke instruments generated C and
test stubs, retaining prebuilt runtime/support libraries and mimalloc with leak
detection disabled; fake resource counters independently balance.

The execution owner can also borrow already loaded weights, which lets the
software ABI fixture exercise complete startup/request/close without streaming
69 GB of synthetic data. That test double never computes model logits. Real
checkpoint upload, NVCC/driver compatibility, model numerics and Spark timing
remain target qualification work.
