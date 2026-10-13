///|
/// counts-dependent separate-weight lowering of the same staged BF16 SwiGLU
/// law. Each row/inner computes gate/up once into a shared product frame;
/// down columns consume that frame instead of repeating the gate/up fold.
pub fn staged_separate_swiglu_bf16_source(
  hidden_size~ : Int,
  intermediate_size~ : Int,
  maximum_rows~ : Int,
  function_prefix~ : String,
  publication? : @precision.ProjectionPublication = LocalBFloat16,
) -> Bytes raise GatedMlpCudaSourceError {
  guard hidden_size > 0 && intermediate_size > 0 else { raise InvalidWidth }
  guard maximum_rows > 0 else { raise InvalidRowCount }
  guard maximum_rows.to_int64() * hidden_size.to_int64() <= 2147483647L &&
    maximum_rows.to_int64() * intermediate_size.to_int64() <= 2147483647L else {
    raise DimensionOverflow
  }
  guard valid_symbol(function_prefix) else { raise InvalidFunctionSymbol }
  let h = hidden_size.to_string()
  let m = intermediate_size.to_string()
  let r = maximum_rows.to_string()
  let plan = @precision.DenseSwiGluPrecision::new(
    hidden_width=hidden_size,
    intermediate_width=intermediate_size,
    maximum_rows~,
  ) catch {
    _ => raise DimensionOverflow
  }
  fn expression(
    op : @precision.DenseSwiGluExpression,
    bindings : ArrayView[String],
  ) -> String {
    @arithmetic.scalar_expression(plan.expression(op), bindings) catch {
      _ => abort("invalid dense program")
    }
  }
  let gate_update = expression(ProjectionUpdate, [
    "gate_sum",
    "input[row*\{h}ULL+column]",
    "gate_weight[inner*\{h}ULL+column]",
  ])
  let up_update = expression(ProjectionUpdate, [
    "up_sum",
    "input[row*\{h}ULL+column]",
    "up_weight[inner*\{h}ULL+column]",
  ])
  let gate_publish = expression(ProjectionPublish, ["gate_sum"])
  let up_publish = expression(ProjectionPublish, ["up_sum"])
  let activation = expression(Activation, ["gate"])
  let product = expression(Product, ["activated", "up"])
  let down_fold = plan.projection_fold(down=true) catch {
    _ => abort("invalid dense fold")
  }
  let down_update = @arithmetic.scalar_expression(down_fold.update, [
    "sum",
    "product[row*\{m}ULL+inner]",
    "down_weight[out*\{m}ULL+inner]",
  ]) catch {
    _ => abort("invalid dense bindings")
  }
  let partial = publication == ReduceFloat32BeforeRound
  let down_publish = if partial {
    "sum"
  } else {
    expression(ProjectionPublish, ["sum"])
  }
  let output_type = if partial { "float" } else { "__nv_bfloat16" }
  let publish = if partial {
    let text =
      $|extern "C" __global__ void \{function_prefix}_publish(const int *counts,const float *partial,__nv_bfloat16 *output) {
      $|  const unsigned long long i=(unsigned long long)blockIdx.x*256+threadIdx.x;
      $|  const int live=counts[3]; if(live<=0||live>\{r}||i>=(unsigned long long)live*\{h}) return;
      $|  output[i]=__float2bfloat16_rn(partial[i]);
      $|}
    text
  } else {
    ""
  }
  @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_gate_up(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *gate_weight, const __nv_bfloat16 *up_weight, __nv_bfloat16 *product) {\n" +
    "const int32_t rows=counts[3]; if(rows<=0 || rows>" +
    r +
    ") return;\n" +
    "const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x; if(i>=(unsigned long long)rows*" +
    m +
    "ULL) return;\n" +
    "const unsigned long long row=i/" +
    m +
    "ULL, inner=i%" +
    m +
    "ULL; float gate_sum=0.0f, up_sum=0.0f;\n" +
    "for(int column=0; column<" +
    h +
    "; ++column) { gate_sum=\{gate_update}; up_sum=\{up_update}; }\n" +
    "const __nv_bfloat16 gate=\{gate_publish}, up=\{up_publish}; const __nv_bfloat16 activated=\{activation}; product[i]=\{product}; }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_down(const int32_t *counts, const __nv_bfloat16 *product, const __nv_bfloat16 *down_weight, \{output_type} *output) {\n" +
    "const int32_t rows=counts[3]; if(rows<=0 || rows>" +
    r +
    ") return;\n" +
    "const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x; if(i>=(unsigned long long)rows*" +
    h +
    "ULL) return;\n" +
    "const unsigned long long row=i/" +
    h +
    "ULL, out=i%" +
    h +
    "ULL; float sum=0.0f; for(int inner=0; inner<" +
    down_fold.extent.to_string() +
    "; ++inner) sum=\{down_update}; output[i]=\{down_publish}; }\n" +
    publish,
  )
}
