///|
test "retained residual emission preserves tree and exports paired physical realizations" {
  for width in [127, 128, 1024, 1025, 2048] {
    let program = @elementwise_physical_ir.NormalizationProgram::new(
      width~,
      owners=128,
      subgroup_lanes=32,
      residual=true,
      owner_tree=true,
    ).unwrap()
    let retained = residual_program_source(program)
    assert_eq(retained.split("__syncthreads()").length(), 2)
    assert_false(retained.contains("__bfloat162float(residual_output"))
    assert_true(retained.contains("partial[0] += partial[2]"))
    if @env.get_env_var("LUNA_EXPORT_RETAINED_NORM") == Some("1") {
      if width == 1024 {
        let fixture = fused_test_fixture()
        let parameters = {
          ..fused_test_residual_norm_candidate(fixture).parameters,
          width,
          epsilon: 0.000001,
        }
        let profile = @launch_contract.PagedKernelProfile::new(
          id=@launch_contract.KernelProfileId::from_stable_int(7),
          max_query_rows=32,
          max_query_tokens=2048,
          max_page_table_entries=8192,
        )
        println("LUNA_SOURCE_BEGIN norm-production")
        println(
          @utf8.decode(
            fused_residual_rmsnorm_production_source(
              "lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2",
              parameters, profile, program,
            ),
          ),
        )
        println("LUNA_SOURCE_END")
      }
      for
        (name, body) in [
          ("old", residual_shared_tree_source(program)),
          ("new", retained),
        ] {
        println("LUNA_SOURCE_BEGIN norm-\{width}-\{name}")
        println(
          "#include <cuda_bf16.h>\n#include <stdint.h>\n#define LF_WIDTH \{width}\n#define LF_BLOCK_THREADS 128\n#define LF_EPSILON 0.000001f",
        )
        println(
          "extern \"C\" __global__ void lf_norm(const __nv_bfloat16 *residual, const __nv_bfloat16 *branch, const __nv_bfloat16 *norm_weight, __nv_bfloat16 *residual_output, __nv_bfloat16 *norm_output) {\nconst int token = blockIdx.x;",
        )
        println(body)
        println("}\nLUNA_SOURCE_END")
      }
    }
  }
  let wide = @elementwise_physical_ir.NormalizationProgram::new(
    width=2049,
    owners=128,
    subgroup_lanes=32,
    residual=true,
    owner_tree=true,
  ).unwrap()
  assert_true(residual_program_source(wide).contains("__syncthreads()"))
}
