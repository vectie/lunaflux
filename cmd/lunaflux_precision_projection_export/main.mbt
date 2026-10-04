///|
fn bytes_initializer(bytes : Bytes) -> String {
  bytes.to_array().map(x => x.to_uint().to_string()).join(",")
}

///|
async fn main {
  guard @env.args() is [_, output_root] else { fail("NEW_OUTPUT_DIRECTORY") }
  if @fs.exists(output_root) {
    fail("no overwrite")
  }
  @fs.mkdir(output_root)
  let shape = @strategy.ProjectionShape::new(
    Dense,
    input_width=128,
    output_width=64,
  )
  let dispatch = @strategy.plan_projection_dispatch(
    shape,
    32,
    @strategy.ProjectionCapabilities::new(
      subgroup_width=32,
      matrix_tile=Some(
        @strategy.ProjectionTile::new(rows=16, columns=16, reduction=16),
      ),
    ),
  )
  let request = @compiler.ProjectionTileCompilerRequest::new(
    dispatch~,
    maximum_rows=32,
    maximum_tokens=32,
    demand=AllTokenRows,
    single_row_numerics=AllowDeterministicDotTree,
  )
  let physical = @compiler.compile_projection_tile(request)
    .refine_physical(
      width => {
        @physical.OperandLayout::new(
          columns=width,
          vector=8,
          rotations=2,
          row_group=1,
        )
      },
      _ => OverlappedTransfer,
      owners_per_group=32,
    )
    .unwrap()
  let matrix = physical.matrix.unwrap()
  let (_, tile_rows, tile_columns) = matrix.operands
    .region(1)
    .unwrap()
    .geometry()
  let tile_elements = tile_rows * tile_columns * matrix.schedule.stage_count()
  let accuracy = @precision.Accuracy::new(absolute=0.1, relative=0.1)
  let activation = @precision.plan_operand(
    role=Activation,
    representation=@format.Representation::new(
      format=Bf16,
      rows=32,
      columns=128,
    ),
    placement=NativeOperand,
    supported_placements=[NativeOperand],
    accuracy~,
    memory_ceiling=1000000L,
  )
  let output = @precision.plan_operand(
    role=Output,
    representation=@format.Representation::new(format=Bf16, rows=32, columns=64),
    placement=NativeOperand,
    supported_placements=[NativeOperand],
    accuracy~,
    memory_ceiling=1000000L,
  )
  let code = StringBuilder()
  let fixtures = StringBuilder()
  fixtures.write_string(
    "#pragma once\n#include <cstring>\nstatic float lf_host_float(unsigned bits) { float v; std::memcpy(&v,&bits,4); return v; }\n",
  )
  code.write_string("#pragma once\n")
  fixtures.write_string(
    "#pragma once\nstruct ProjectionCase { const char *name; int payload_bytes,scale_bytes; unsigned global_bits; const unsigned char *payload,*scales; const float *expected; void (*launch)(const int*,const __nv_bfloat16*,const unsigned char*,const unsigned char*,const float*,__nv_bfloat16*); };\n",
  )
  let cases = []
  for
    format in [
      @format.Format::Bf16,
      Fp16,
      F32,
      Fp8E4M3,
      Fp8E5M2,
      Int8,
      Int4,
      Fp4E2M1,
      NvFp4,
      MxFp4,
      MxFp8E4M3,
      MxFp8E5M2,
    ] {
    let i = cases.length()
    let symbol = "lf_compact_projection_" + i.to_string()
    let rep = @format.Representation::new(format~, rows=64, columns=128)
    let values = Array::makei(64 * 128, i => {
      Float::from_int(i % 17 - 8) * 0.125F
    })
    let packed = @format.quantize(rep, values[:])
    let weight = @precision.plan_operand(
      role=Weight,
      representation=rep,
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy~,
      memory_ceiling=1000000L,
      tile_elements~,
    )
    let numerical = @precision.LinearPrecision::new(
      activation~,
      weight~,
      output~,
      route=DecodeToBf16,
      accumulator=F32,
    )
    let source = @projection.precision_projection_source(
      physical,
      numerical,
      symbol~,
    )
    let (grid, threads) = source.launch()
    code.write_string(@utf8.decode(source.source_bytes()) + "\n")
    let launch =
      $|static void \{symbol}_launch(const int *counts,const __nv_bfloat16 *input,const unsigned char *weight,const unsigned char *scales,const float *global,__nv_bfloat16 *output) {
      $|  \{symbol}<<<\{grid},\{threads}>>>(counts,input,weight,scales,global,output);
      $|}
      $|
    code.write_string(launch)
    let expected = Array::makei(32 * 64, i => {
      let row = i / 64
      let column = i % 64
      for k in 0..<128; sum = 0.0F {
        let input = Float::from_int((row * 128 + k) % 13 - 6) * 0.125F
        let decoded = @format.decode_element(
          Bf16,
          @format.encode_element(Bf16, packed.decode(column, k)),
        )
        continue sum + input * decoded
      } nobreak {
        sum
      }
    })
    let expect = expected
      .map(x => "lf_host_float(\{x.reinterpret_as_uint()}U)")
      .join(",")
    fixtures.write_string(
      "static const unsigned char \{symbol}_payload[]={\{bytes_initializer(packed.payload())}};\n",
    )
    fixtures.write_string(
      "static const unsigned char \{symbol}_scales[]={\{bytes_initializer(packed.scales())}};\n",
    )
    fixtures.write_string(
      "static const float \{symbol}_expected[]={\{expect}};\n",
    )
    cases.push(
      "{\"\{format.name()}\",\{rep.payload_bytes()},\{rep.scale_bytes()},\{packed.global_scale().reinterpret_as_uint()}U,\{symbol}_payload,\{symbol}_scales,\{symbol}_expected,\{symbol}_launch}",
    )
  }
  fixtures.write_string(
    "static const ProjectionCase projection_cases[]={" +
    cases.join(",\n") +
    "};\n",
  )
  @fs.write_file(
    output_root + "/precision_kernels.cuh",
    code.to_string(),
    create_mode=CreateNew,
  )
  @fs.write_file(
    output_root + "/precision_fixture.h",
    fixtures.to_string(),
    create_mode=CreateNew,
  )
  @fs.write_file(
    output_root + "/REPRESENTATIONS.txt",
    "scope=all-format-compact-to-BF16-projection-not-native-low-bit-MMA\n",
    create_mode=CreateNew,
  )
  println("exported=12 compact projection kernels through existing physical IR")
}
