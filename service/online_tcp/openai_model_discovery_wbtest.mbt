///|
test "model discovery uses admitted UTF8 alias and context with exact byte length" {
  let name = "模型\"\\one"
  let encoded = @utf8.encode("_" + name + "_")
  let alias = FixedArray::makei(encoded.length(), i => encoded[i])
  let bytes = luna_openai_model_response(alias, 1, encoded.length() - 2, 2048)
  let response = @utf8.decode(bytes)
  let pieces = response.split("\r\n\r\n").collect()
  assert_eq(pieces.length(), 2)
  let body = pieces[1].to_string()
  assert_true(
    pieces[0].contains("Content-Length: \{@utf8.encode(body).length()}"),
  )
  assert_true(body.contains("\"max_model_len\":2048"))
  assert_true(body.contains("模型\\\"\\\\one"))
  let control = new_luna_openai_embedded_control(models_response=bytes)
  assert_true(control.begin())
  let head = @utf8.encode("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
  let input = FixedArray::makei(head.length(), i => head[i])
  assert_eq(control.offer(input, 0, 9), 0)
  assert_eq(control.offer(input, 9, input.length() - 9), 5)
  assert_true(control.prepare_models())
  assert_eq(control.output, bytes)
  control.clear()
  assert_true(control.begin())
  control.clear()
}

///|
test "unbound control does not invent a model discovery listing" {
  let control = new_luna_openai_embedded_control()
  assert_true(control.prepare_models())
  assert_true(@utf8.decode(control.output).has_prefix("HTTP/1.1 400"))
}
