///|
fn bytes_initializer(bytes : Bytes) -> String {
  if bytes.is_empty() {
    "0"
  } else {
    bytes.to_array().map(x => x.to_uint().to_string()).join(",")
  }
}

///|
fn float_initializer(value : Float) -> String {
  let repr = value.to_string()
  if repr.contains(".") || repr.contains("e") || repr.contains("E") {
    repr + "f"
  } else {
    repr + ".0f"
  }
}

///|
async fn main {
  guard @env.args() is [_, output] else { fail("NEW_OUTPUT_DIRECTORY") }
  if @fs.exists(output) {
    fail("no overwrite")
  }
  @fs.mkdir(output)
  let accuracy = @precision.Accuracy::new(absolute=0.2, relative=0.1)
  let code = StringBuilder()
  code.write_string("#pragma once\n")
  let fixture = StringBuilder()
  fixture.write_string(
    "#pragma once\n#include <cuda_bf16.h>\nstruct PrecisionCase { const char *name; int rows,columns,payload_bytes,scale_bytes; unsigned global_bits; const float *input; const unsigned char *payload,*scales,*expected; void (*launch)(const float*,unsigned char*,unsigned char*,float*,__nv_bfloat16*,int*); };\n",
  )
  let cases = []
  let identities = []
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
    for (rows, columns) in [(1, 1), (3, 35), (2, 257)] {
      let index = cases.length()
      let symbol = "lf_precision_" + index.to_string()
      let rep = @format.Representation::new(format~, rows~, columns~)
      let input = Array::makei(rows * columns, i => {
        let code = i % 23 - 11
        let magnitude = Float::from_int(code) * 0.25F
        if i % 41 == 0 {
          -0.0F
        } else {
          magnitude
        }
      })
      let packed = @format.quantize(rep, input[:])
      let expanded = FixedArray::make(rows * columns * 2, b'\x00')
      packed.decode_bf16_chunk(0L, rows * columns, expanded)
      let quant = @precision.plan_operand(
        role=Activation,
        representation=rep,
        placement=QuantizeBeforeConsumer,
        supported_placements=[QuantizeBeforeConsumer],
        accuracy~,
        memory_ceiling=1048576L,
      )
      let decode = @precision.plan_operand(
        role=Weight,
        representation=rep,
        placement=StartupExpandBf16,
        supported_placements=[StartupExpandBf16],
        accuracy~,
        memory_ceiling=1048576L,
      )
      let quant_source = @cuda_precision.quantize_source(quant, symbol~)
      let decode_source = @cuda_precision.decode_source(
        decode,
        symbol=symbol + "_decode",
      )
      code.write_string(
        quant_source.source() + "\n" + decode_source.source() + "\n",
      )
      identities.push(
        rep.canonical() + "|decode-source=" + decode_source.digest(),
      )
      let launches = StringBuilder()
      for index, stage in quant_source.stages() {
        let (grid, block) = quant_source.launches()[index].dimensions()
        launches.write_string(
          match stage {
            GlobalScale =>
              "\{symbol}_global<<<\{grid},\{block}>>>(input,global,error);"
            BlockScales =>
              "\{symbol}_scales<<<\{grid},\{block}>>>(input,scales,global,error);"
            Pack =>
              "\{symbol}_pack<<<\{grid},\{block}>>>(input,payload,scales,global,error);"
          },
        )
      }
      let (decode_grid, decode_block) = decode_source.launch().dimensions()
      let launch =
        $|static void \{symbol}_launch(const float *input,unsigned char *payload,unsigned char *scales,float *global,__nv_bfloat16 *out,int *error) {
        $|\{launches.to_string()}
        $|\{symbol}_decode<<<\{decode_grid},\{decode_block}>>>(payload,scales,global,out,error);
        $|}
        $|
      code.write_string(launch)
      let inputs = input
        .map(x => {
          if x.reinterpret_as_uint() == 0x80000000U {
            "-0.0f"
          } else {
            float_initializer(x)
          }
        })
        .join(",")
      fixture.write_string(
        "static const float \{symbol}_input[]={\{inputs}};\n",
      )
      fixture.write_string(
        "static const unsigned char \{symbol}_payload[]={\{bytes_initializer(packed.payload())}};\n",
      )
      fixture.write_string(
        "static const unsigned char \{symbol}_fixture_scales[]={\{bytes_initializer(packed.scales())}};\n",
      )
      fixture.write_string(
        "static const unsigned char \{symbol}_expected[]={\{bytes_initializer(Bytes::from_array(expanded[:]))}};\n",
      )
      cases.push(
        "{\"\{format.name()}-\{rows}x\{columns}\",\{rows},\{columns},\{rep.payload_bytes()},\{rep.scale_bytes()},\{packed.global_scale().reinterpret_as_uint()}U,\{symbol}_input,\{symbol}_payload,\{symbol}_fixture_scales,\{symbol}_expected,\{symbol}_launch}",
      )
    }
  }
  fixture.write_string(
    "static const PrecisionCase precision_cases[]={\n" +
    cases.join(",\n") +
    "\n};\n",
  )
  @fs.write_file(
    output + "/precision_kernels.cuh",
    code.to_string(),
    create_mode=CreateNew,
  )
  @fs.write_file(
    output + "/precision_fixture.h",
    fixture.to_string(),
    create_mode=CreateNew,
  )
  @fs.write_file(
    output + "/REPRESENTATIONS.txt",
    identities.join("\n") + "\n",
    create_mode=CreateNew,
  )
  println("exported=\{cases.length()} exact-scalar-reference-cases")
}
