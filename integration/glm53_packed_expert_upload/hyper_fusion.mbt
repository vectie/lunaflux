///|
/// Coordinates refer to the unmodified launch sequence. Each envelope begins
/// with its square reduction and ends with its residual publication.
fn GlmDecoderLayer::hyper_sites(
  self : Self,
) -> Array[(Int, Int, @hyper.HyperConnectionFrame)] raise {
  let (attention, ffn) = match self {
    Recurrent(layer) => (layer.hyper, layer.ffn)
    Sparse(layer) => (Some(layer.decoder.hyper_envelope()), Some(layer.ffn))
  }
  guard attention is Some(attention) && ffn is Some(ffn) else { return [] }
  let end = self.launches().length()
  let split = end - ffn.launches.length()
  [(0, split - 1, attention), (split, end - 1, ffn.envelope)]
}

///|
fn fuse_hyper_sites(
  launches : ArrayView[@device.OrderedKernelLaunch],
  sites : ArrayView[(Int, Int, @hyper.HyperConnectionFrame)],
  aot : @device.Module,
  post_square~ : Bool,
  control_tail~ : Bool,
  deferred~ : Bool,
) -> (Array[@device.OrderedKernelLaunch], @parallel.AdjacentFusion) raise {
  let spans = []
  let fused = []
  for i, (square, _, consumer) in sites {
    if (post_square || deferred) && i > 0 {
      let (_, post, producer) = sites[i - 1]
      guard post + 1 == square else {
        raise @precision_ir.PrecisionPlanError::IllegalPlacement
      }
      spans.push((post, if deferred { 3 } else { 2 }))
      fused.push(
        if deferred {
          producer.fuse_deferred_post_pre(consumer, aot)
        } else {
          producer.fuse_post_square(consumer, aot)
        },
      )
    }
    if control_tail {
      spans.push((square + 2, 2))
      fused.push(consumer.fuse_control_tail(aot))
    }
  }
  let rewrite = @parallel.AdjacentFusion::spans(
    kernel_count=launches.length(),
    spans,
  )
  let result = []
  let mut pair = 0
  let mut index = 0
  while index < launches.length() {
    if pair < spans.length() && index == spans[pair].0 {
      result.push(fused[pair])
      index += spans[pair].1
      pair += 1
    } else {
      result.push(launches[index])
      index += 1
    }
  }
  (result, rewrite)
}
