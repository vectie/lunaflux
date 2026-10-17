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
) -> (Array[@device.OrderedKernelLaunch], @parallel.AdjacentFusion) raise {
  let pairs = []
  let fused = []
  for i in 1..<sites.length() {
    let (_, post, producer) = sites[i - 1]
    let (square, _, consumer) = sites[i]
    guard post + 1 == square else {
      raise @precision_ir.PrecisionPlanError::IllegalPlacement
    }
    pairs.push(post)
    fused.push(producer.fuse_post_square(consumer, aot))
  }
  let rewrite = @parallel.AdjacentFusion::new(
    kernel_count=launches.length(),
    pairs,
  )
  let result = []
  let mut pair = 0
  let mut index = 0
  while index < launches.length() {
    if pair < pairs.length() && index == pairs[pair] {
      result.push(fused[pair])
      pair += 1
      index += 2
    } else {
      result.push(launches[index])
      index += 1
    }
  }
  (result, rewrite)
}
