# Scalar and register-window physical plans

The non-pipelined projection routes refine the existing semantic/schedule
choice into immutable physical traversal descriptors before CUDA text emission.
They do not switch precision, numerical permissions, or runtime dispatch.

- `ProjectionScalarFoldPlan` assigns one shared left operand, independent right
  operands, accumulator IDs, exact strided component ownership, and the ordered
  partial-sum tree. Ordered mode has one worker and no tree. Products and sums
  continue to round separately; purity does not authorize reassociation.
- `ProjectionOperandWindow` assigns initial and subsequent operand values to
  a bounded ahead-of-use window. `read_at` rejects the final partial window's
  missing members rather than inserting zero-product reductions. Selected-row
  vocabulary projection consumes these values in increasing reduction order.
- `mapped_fragment_program` refines the schedule's reduction loop into the
  shared physical fragment dialect. Resident selected rows extend the strip,
  not the operation's reduction semantics. Unsupported device instruction
  shapes are rejected by the CUDA realization, not by this generic adapter.

The CUDA backend owns thread/lane maps, register encodings, `mma`/shuffle
instructions, and selected-row tensor address bindings. No scheduler or model
family gains an architecture-dependent branch. These plans are offline
compiler values; they do not allocate or validate on the token execution path.

```mbt check
///|
test {
  let plan = @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
    extent=65,
    products=2,
    StridedPairwiseDot(32),
  ).unwrap()
  assert_eq(plan.read_at(worker=0, iteration=2), Some(64))
  assert_eq(plan.read_at(worker=1, iteration=2), None)
  assert_eq(plan.product(1), Some((0, 2, 1)))
}
```
