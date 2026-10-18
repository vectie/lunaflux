///|
/// Destination ownership is transposed only in private registers. Publication
/// and the following square fold retain the original flattened stripe order.
fn deferred_component_source(
  producer : @precision.HyperConnectionPrecision,
  hidden : Int,
  streams : Int,
  workers : Int,
) -> String raise {
  let law = producer.mix_law()
  let branch = @arithmetic.scalar_expression(law.expression(BranchProduct), [
    "post_coeff[destination]", "branch_value",
  ])
  let update = @arithmetic.scalar_expression(law.expression(ResidualUpdate), [
    "residual_sum", "mix_coeff[source][destination]", "source_value[source]",
  ])
  let publish = @arithmetic.scalar_expression(law.expression(Publish), [
    "branch_product", "residual_sum",
  ])
  (
    $| float post_coeff[\{streams}], mix_coeff[\{streams}][\{streams}];
    $| #pragma unroll
    $| for(int d=0;d<\{streams};++d) {
    $|   post_coeff[d]=post[row*\{streams}+d];
    $|   #pragma unroll
    $|   for(int s=0;s<\{streams};++s) mix_coeff[s][d]=combine[(row*\{streams}+s)*\{streams}+d];
    $| }
    $| #pragma unroll
    $| for(int i=0;i<\{hidden/workers};++i) {
    $|   const int component=worker+i*\{workers};
    $|   const float branch_value=__bfloat162float(branch[row*\{hidden}+component]);
    $|   float source_value[\{streams}];
    $|   #pragma unroll
    $|   for(int s=0;s<\{streams};++s) source_value[s]=__bfloat162float(residual[(row*\{streams}+s)*\{hidden}+component]);
    $|   #pragma unroll
    $|   for(int destination=0;destination<\{streams};++destination) {
    $|     const float branch_product=\{branch}; float residual_sum=0.0f;
    $|     #pragma unroll
    $|     for(int source=0;source<\{streams};++source) residual_sum=\{update};
    $|     const __nv_bfloat16 rounded=\{publish};
    $|     const int column=destination*\{hidden}+component;
    $|     if(tile==0) output[row*\{streams*hidden}+column]=rounded;
    $|     values[destination*\{hidden/workers}+i]=__bfloat162float(rounded);
    $|   }
    $| }
  )
}
