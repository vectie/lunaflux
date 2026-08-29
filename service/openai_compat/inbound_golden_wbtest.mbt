///|
test "Luna OpenAI inbound Responses is split-budget stable and canonical" {
  let inference = openai_wb_fixture_limits()
  let body = "{ \"stream\" : true, \"input\" : \"hi \\uD83D\\uDE00\", \"model\" : \"model-x\" }"
  let mut baseline : FixedArray[Byte] = []
  let mut baseline_total = 0UL
  for budget in [1, 17, 65536] {
    let (actual, total) = inbound_encode(
      inbound_workspace(inference, budget),
      inbound_http_view(1, body),
    )
    assert_eq(inbound_text(actual, inference), "hi 😀")
    if baseline.length() == 0 {
      baseline = actual
      baseline_total = total
      assert_eq(total, 111UL)
    } else {
      openai_wb_assert_same(actual, baseline)
      assert_eq(total, baseline_total)
    }
  }
}

///|
test "Luna OpenAI inbound Chat uses explicit template and arbitrary key order" {
  let inference = openai_wb_fixture_limits()
  let body = "{\"messages\":[" +
    "{\"content\":\"rules\",\"role\":\"system\"}," +
    "{\"content\" : \"he\\nllo\",\"role\" : \"user\"}," +
    "{\"role\":\"assistant\",\"content\":\"ok\"}," +
    "{\"content\":\"中\",\"role\":\"user\"}" +
    "],\"stream\": true,\"model\":\"model-x\"}"
  let mut baseline : FixedArray[Byte] = []
  let mut baseline_total = 0UL
  for budget in [1, 17, 65536] {
    let (actual, total) = inbound_encode(
      inbound_workspace(inference, budget),
      inbound_http_view(2, body),
    )
    assert_eq(
      inbound_text(actual, inference),
      "<S>rules</S><U>he\nllo</U><A>ok</A><U>中</U><A>",
    )
    if baseline.length() == 0 {
      baseline = actual
      baseline_total = total
      assert_eq(total, 273UL)
    } else {
      openai_wb_assert_same(actual, baseline)
      assert_eq(total, baseline_total)
    }
  }
}

///|
test "Luna OpenAI inbound accepts escaped model and four-byte raw UTF-8" {
  let inference = openai_wb_fixture_limits()
  let body = "{\"input\":\"😀\",\"model\":\"model\\u002dx\",\"stream\":true}"
  let (frame, _) = inbound_encode(
    inbound_workspace(inference, 1),
    inbound_http_view(1, body),
  )
  assert_eq(inbound_text(frame, inference), "😀")
}
