///|
test "producer consumer slots preserve publication and retirement across tails" {
  for stages in [2, 3, 8] {
    let p = @physical_tile_ir.ProducerConsumerPipeline::new(
      stages~,
      producers=128,
      consumers=128,
    ).unwrap()
    assert_eq(p.geometry(), (stages, 128, 128))
    for count in [1, 2, 3, 17] {
      let retired = Array::make(stages, -1)
      for epoch in 0..<count {
        let slot = p.slot(epoch).unwrap()
        if p.reuse_after(epoch) is Some(previous) {
          assert_eq(retired[slot], previous)
        }
        retired[slot] = epoch
      }
    }
    assert_eq(p.slot(-1), None)
  }
  assert_eq(
    @physical_tile_ir.ProducerConsumerPipeline::new(
      stages=1,
      producers=128,
      consumers=128,
    ),
    None,
  )
}
