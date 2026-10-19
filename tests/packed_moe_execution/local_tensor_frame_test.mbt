///|
test "locally combined TP borrows BF16 output after both producers and before HC" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let module = context.load_module(b"fake\x00")
  defer (module.close() catch { _ => () })
  let (program, routed, shared) = routed_program_fixture(
    tensor=true,
    reduction_placement=LocalCombinedBFloat16,
  )
  defer (program.close() catch { _ => () })
  let buffers = [20L, 128L, 256L, 16L, routed, shared, 128L].map(n => {
    context.allocate(n)
  })
  defer (for b in buffers.rev() {
    b.close() catch {
      _ => ()
    }
  })
  let boundary = program.prepare_routed_tensor_frame(
    context,
    module,
    maximum_workspace_bytes=program.workspace_bytes(),
    counts=buffers[0],
    input=buffers[1],
    router_weight=buffers[2],
    correction_bias=buffers[3],
    routed_bank=buffers[4],
    shared_bank=buffers[5],
    output=buffers[6],
  )
  assert_eq(boundary.kind(), Bf16SumAllReduce)
  assert_eq(boundary.width(), 32)
  assert_true(boundary.partials().same_storage(buffers[6]))
  assert_eq(boundary.before_kernel(), program.launches().length())
  assert_true(program.shared_tensor_boundary() is None)
  guard program.producer_fork() is Some(fork) else { fail("missing join") }
  assert_eq(fork.range().2, boundary.before_kernel() - 1)
  assert_true(fork.range().1 > 0)
  program.close()
  for b in buffers.rev() {
    b.close()
  }
  module.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
