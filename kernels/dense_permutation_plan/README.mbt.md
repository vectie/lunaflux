# Dense permutation planning

A backend-neutral immutable map from output-linear to input-linear indices.
Positive dense shapes of rank 1–8 are bounded by an explicit element ceiling.
Axes of extent one disappear; adjacent axes contiguous in both layouts merge.
No tensor-sized index table is stored. Inversion supports pack/unpack using the
same semantics. The plan does not allocate device memory or prescribe a tile.

```mbt check
///|
test "transpose scalar mapping" {
  let plan = @dense_permutation_plan.Permutation::new(
    [2, 3],
    [1, 0],
    max_elements=6L,
  )
  assert_eq(plan.source_index(1L), 3L)
  assert_eq(plan.inverse().source_index(3L), 1L)
}
```
