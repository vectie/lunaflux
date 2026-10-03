///|
test "async decode consumes static retained copy ownership without indexed arrays" {
  for id in [440, 441, 452, 453, 454, 455, 460, 461] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=id >= 450,
    )
    let retention = lowering.decode_copy_retention().unwrap()
    let (vectors, owners, slots) = retention.dimensions()
    let elements = lowering.decode_copy_elements()
    assert_eq(vectors, lowering.tile().1 * lowering.tile().2 / elements)
    assert_eq(elements, if id >= 450 { 8 } else { 4 })
    assert_eq(owners, lowering.launch().2)
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "copy_retention",
        1048576,
      ).bytes(),
    )
    assert_true(
      source.contains(
        "Retained copy ownership: \{vectors} vectors, \{owners} owners, \{slots} scalar slots",
      ),
    )
    assert_false(source.contains("addresses[copies]"))
    assert_false(source.contains("sizes[copies]"))
    assert_false(source.contains("addresses[item]"))
    assert_false(source.contains("sizes[item]"))
    assert_true(source.contains("[%0], [%1], \{elements * 2}, %2;"))
    let independent = match lowering.physical_program() {
      BlockwiseKeyProgram(_, effects) => effects.independently_refills()
      _ => false
    }
    for slot in 0..<slots {
      assert_true(
        source.contains("const int vector = thread + \{slot * owners}"),
      )
      if !independent {
        assert_true(
          source.contains("unsigned long long retained_address_\{slot} = 0"),
        )
        assert_true(source.contains("value_cache + retained_address_\{slot}"))
        assert_true(source.contains("\"r\"(retained_size_\{slot})"))
      }
    }
    if independent {
      assert_true(source.contains("future_origin_0 = 0, future_origin_1 = 0"))
      assert_true(source.contains("future_valid_0 = 0, future_valid_1 = 0"))
      assert_true(source.contains("value_cache + cache_index"))
      assert_false(source.contains("retained_address_"))
    }
  }
  assert_eq(
    source_test_decode_lowering(candidate_id=430).decode_copy_retention(),
    None,
  )
}
