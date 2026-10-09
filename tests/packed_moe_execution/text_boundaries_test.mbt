///|
fn boundary_text_frame(
  boundary : @precision.TextBoundary,
) -> @text_frame.TextIoFrame raise {
  @text_frame.TextIoFrame::new(
    @text_source.text_io_source(
      @precision.TextIoPrecision::new(
        vocabulary_size=11,
        hidden_width=6,
        stream_count=1,
        maximum_rows=4,
        maximum_output_rows=2,
        normalization_epsilon=1.0e-5,
        maximum_workspace_bytes=112L,
      ),
      symbol_prefix="text_boundary_fixture",
    ),
    boundary~,
  )
}

///|
test "pipeline text boundaries bind only their own kernels and workspace" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let ingress = boundary_text_frame(Ingress)
  defer (ingress.close() catch { _ => () })
  let egress = boundary_text_frame(Egress)
  defer (egress.close() catch { _ => () })
  let interior = boundary_text_frame(Interior)
  defer (interior.close() catch { _ => () })
  assert_eq(ingress.workspace_bytes(), 0L)
  assert_eq(interior.workspace_bytes(), 0L)
  assert_eq(egress.workspace_bytes(), 112L)
  ingress.prepare_ingress(
    aot,
    counts=operand,
    token_ids=operand,
    embedding=operand,
    input_residual=operand,
  )
  egress.prepare_egress(
    context,
    aot,
    counts=operand,
    output_count=operand,
    selected_rows=operand,
    output_residual=operand,
    norm_weight=operand,
    head_weight=operand,
    sampled_tokens=operand,
    maximum_workspace_bytes=112L,
  )
  interior.prepare_interior()
  assert_eq(ingress.prefix().length(), 1)
  assert_eq(ingress.suffix().length(), 0)
  assert_eq(egress.prefix().length(), 0)
  assert_eq(egress.suffix().length(), 3)
  assert_eq(interior.prefix().length(), 0)
  assert_eq(interior.suffix().length(), 0)
  assert_eq(egress.logits().byte_count(), 88L)
  try ingress.logits() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("ingress exposed nonexistent logits")
  }
  interior.close()
  egress.close()
  ingress.close()
  operand.close()
  aot.close()
  context.close()
  assert_eq(live_resources(), baseline)
}

///|
test "wrong boundary or undersized egress owns no new resources" {
  let context = @device.open_context(0)
  defer (context.close() catch { _ => () })
  let aot = context.load_module(b"fake\x00")
  defer (aot.close() catch { _ => () })
  let operand = context.allocate(256L)
  defer (operand.close() catch { _ => () })
  let baseline = live_resources()
  for boundary in [@precision.TextBoundary::Ingress, Egress] {
    let frame = boundary_text_frame(boundary)
    defer frame.close()
    try
      frame.prepare_egress(
        context,
        aot,
        counts=operand,
        output_count=operand,
        selected_rows=operand,
        output_residual=operand,
        norm_weight=operand,
        head_weight=operand,
        sampled_tokens=operand,
        maximum_workspace_bytes=111L,
      )
    catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      e => raise e
    } noraise {
      _ => fail("wrong boundary/budget accepted")
    }
    assert_eq(live_resources(), baseline)
  }
}
