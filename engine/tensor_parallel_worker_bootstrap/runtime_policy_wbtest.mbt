///|
test "v5 source policy admits exact topology-bound runtime capacity" {
  let topology = tp_bootstrap_topology()
  let runtime = admit_runtime_policy(tp_bootstrap_source(), topology)
  assert_eq(runtime.world_size(), 2)
  assert_eq(runtime.target(), topology.target())
  let rank_zero = runtime.rank_capacity(0).unwrap()
  assert_eq(rank_zero.process_ordinal(), 2)
  assert_eq(rank_zero.memory_ceiling_bytes(), 1073741824L)
  assert_eq(rank_zero.kv_reserved_bytes(), 1048576L)
  assert_eq(rank_zero.collective_reserved_bytes(), 1048576L)
}

///|
test "ordinary source cannot authorize tensor-parallel runtime" {
  let tensor_parallel = tp_bootstrap_source()
  let ordinary = @worker_wire.admit_bootstrap_source(
    tensor_parallel.roots(),
    tensor_parallel.model(),
    tensor_parallel.kernel(),
    tensor_parallel.execution(),
    tensor_parallel.ceilings(),
  )
  try admit_runtime_policy(ordinary, tp_bootstrap_topology()) catch {
    Rejected(rule=WorkerContract, rank=-1) => ()
    _ => fail("unexpected ordinary-source rejection")
  } noraise {
    _ => fail("ordinary source authorized tensor-parallel runtime")
  }
}
