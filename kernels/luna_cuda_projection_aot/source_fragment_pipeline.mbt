///|
/// Lower immutable operand lookahead separately from the ordered arithmetic
/// fold. The two physical register slots are a backend realization of the
/// compiler's bounded fragment lifetime plan, not a model-specific policy.
fn projection_fragment_pipeline_source(
  rows : Int,
  width : Int,
  input_elements : Int,
  stages : Int,
  products? : Array[(Int, String)] = [],
  row_origin? : String = "row_begin",
  row_owner? : String = "consumer_row",
  row_limit? : String = "pipeline_rows",
  column_owner? : String = "consumer_column",
  weight_displacement? : String = "0",
) -> String {
  let products = if products.is_empty() {
    [(input_elements, "c")]
  } else {
    products
  }
  let declarations = StringBuilder()
  let loads = StringBuilder()
  let folds = StringBuilder()
  for index, (offset, accumulator) in products {
    declarations.write_string("unsigned fragments_b\{index}[\{stages}][4];\n")
    let load =
      $|{
      $|  unsigned *b=fragments_b\{index}[slot];
      $|  const unsigned ba=(unsigned)__cvta_generic_to_shared(current+\{offset}+\{weight_displacement}+operand_index(\{column_owner}*16+lane%16,k+lane/16*8));
      $|  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b[0]),"=r"(b[1]),"=r"(b[2]),"=r"(b[3]) : "r"(ba));
      $|}
      $|
    loads.write_string(load)
    let fold =
      $|{
      $|  const unsigned *b=fragments_b\{index}[slot];
      $|  float *d=\{accumulator}[row]+half*4;
      $|  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[half]),"r"(b[half+2]));
      $|}
      $|
    folds.write_string(fold)
  }
  let prime = if stages > 1 { "read_fragment(0,0);" } else { "" }
  let advance = if stages > 1 {
    "if (k+16<\{width}) read_fragment(k+16,(slot+1)%\{stages});"
  } else {
    "read_fragment(k,0);"
  }
  let source =
    $|          unsigned fragments_a[\{stages}][\{rows}][4];
    $|          \{declarations.to_string()}
    $|          auto read_fragment = [&](int k, int slot) {
    $|            \{loads.to_string()}
    $|            #pragma unroll
    $|            for (int row=0;row<\{rows};++row) {
    $|              if (\{row_origin}+(\{row_owner}+row)*16>=\{row_limit}) break;
    $|              unsigned *a=fragments_a[slot][row];
    $|              const unsigned aa=(unsigned)__cvta_generic_to_shared(current+operand_index((\{row_owner}+row)*16+lane%16,k+lane/16*8));
    $|              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(aa));
    $|            }
    $|          };
    $|          \{prime}
    $|          #pragma unroll
    $|          for (int k = 0; k < \{width}; k += 16) {
    $|            const int slot=(k/16)%\{stages};
    $|            \{advance}
    $|            #pragma unroll
    $|            for (int row=0;row<\{rows};++row) {
    $|              if (\{row_origin}+(\{row_owner}+row)*16>=\{row_limit}) break;
    $|              const unsigned *a=fragments_a[slot][row];
    $|              #pragma unroll
    $|              for (int half=0;half<2;++half) {
    $|                \{folds.to_string()}
    $|              }
    $|            }
    $|          }
  source
}
