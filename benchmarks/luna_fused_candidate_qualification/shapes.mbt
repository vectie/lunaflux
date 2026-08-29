///|
pub(all) enum FusedCandidateBenchmarkShape {
  SingleOrigin
  SinglePageTail
  CrossPagePair
  EightTokenPageBoundaries
} derive(Debug, Eq)

///|
fn fused_benchmark_shape_ordinal(shape : FusedCandidateBenchmarkShape) -> Int {
  match shape {
    SingleOrigin => 0
    SinglePageTail => 1
    CrossPagePair => 2
    EightTokenPageBoundaries => 3
  }
}

///|
fn fused_benchmark_shape_name(shape : FusedCandidateBenchmarkShape) -> String {
  match shape {
    SingleOrigin => "single-origin"
    SinglePageTail => "single-page-tail"
    CrossPagePair => "cross-page-pair"
    EightTokenPageBoundaries => "eight-token-page-boundaries"
  }
}

///|
fn fused_benchmark_shape_tokens(shape : FusedCandidateBenchmarkShape) -> Int {
  match shape {
    SingleOrigin | SinglePageTail => 1
    CrossPagePair => 2
    EightTokenPageBoundaries => 8
  }
}

///|
fn fused_benchmark_shapes() -> ReadOnlyArray[FusedCandidateBenchmarkShape] {
  [SingleOrigin, SinglePageTail, CrossPagePair, EightTokenPageBoundaries]
}

///|
fn fused_benchmark_shape_set_bytes() -> Bytes {
  @utf8.encode(
    "schema=lunaflux-fused-scalar-referee-shape-set.v1\n" +
    "shape_count=4\n" +
    "shape=0,single-origin,0,1\n" +
    "shape=1,single-page-tail,15,2\n" +
    "shape=2,cross-page-pair,15|16,1|3\n" +
    "shape=3,eight-token-page-boundaries,0|1|14|15|16|17|30|31,0|2\n",
  )
}

///|
/// Identity of the fixed four-shape Phase-5 microbenchmark matrix. Qualification
/// does not accept caller-defined or widened shape sets.
pub fn fused_candidate_benchmark_shape_set_identity() -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(fused_benchmark_shape_set_bytes()),
    ),
  ) catch {
    _ => abort("SHA-256 produced an invalid fused benchmark shape identity")
  }
}
