///|
fn valid_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 128 &&
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(unit => unit is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise AliasFreeActivationCudaSourceError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise DimensionOverflow
  }
  left * right
}

///|
pub fn f32_alias_free_snake_beta_source(
  batch_count~ : Int,
  channel_count~ : Int,
  time_count~ : Int,
  ratio~ : Int,
  kernel_size~ : Int,
  upsample_symbol~ : String,
  activation_symbol~ : String,
  downsample_symbol~ : String,
) -> AliasFreeActivationCudaSource raise AliasFreeActivationCudaSourceError {
  if batch_count <= 0 ||
    channel_count <= 0 ||
    time_count <= 0 ||
    ratio != 2 ||
    kernel_size != 12 {
    raise InvalidGeometry
  }
  if !valid_symbol(upsample_symbol) ||
    !valid_symbol(activation_symbol) ||
    !valid_symbol(downsample_symbol) ||
    upsample_symbol == activation_symbol ||
    upsample_symbol == downsample_symbol ||
    activation_symbol == downsample_symbol {
    raise InvalidFunctionSymbol
  }
  if time_count > 1073741823 {
    raise DimensionOverflow
  }
  let upsample_time = time_count * ratio
  let input_elements = checked_mul(
    checked_mul(batch_count.to_int64(), channel_count.to_int64()),
    time_count.to_int64(),
  )
  let intermediate_elements = checked_mul(
    checked_mul(batch_count.to_int64(), channel_count.to_int64()),
    upsample_time.to_int64(),
  )
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\nextern \"C\" __global__ void " +
    upsample_symbol +
    "(const float *input,const float *filter,float *upsampled){const unsigned long long flat=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(flat>=" +
    intermediate_elements.to_string() +
    "ULL)return;const int t=(int)(flat%" +
    upsample_time.to_string() +
    "ULL);const unsigned long long outer=flat/" +
    upsample_time.to_string() +
    "ULL;const int full_t=t+15;float sum=0.0f;for(int k=0;k<12;k+=1){const int numerator=full_t-k;if(numerator>=0&&(numerator&1)==0){const int padded=numerator/2;if(padded<" +
    (time_count + 10).to_string() +
    "){int source_t=padded-5;if(source_t<0)source_t=0;if(source_t>=" +
    time_count.to_string() +
    ")source_t=" +
    (time_count - 1).to_string() +
    ";sum=__fadd_rn(sum,__fmul_rn(input[outer*" +
    time_count.to_string() +
    "ULL+(unsigned long long)source_t],filter[k]));}}}upsampled[flat]=__fmul_rn(2.0f,sum);}\nextern \"C\" __global__ void " +
    activation_symbol +
    "(const float *upsampled,const float *alpha,const float *beta,float *activated){const unsigned long long flat=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(flat>=" +
    intermediate_elements.to_string() +
    "ULL)return;const unsigned long long channel=(flat/" +
    upsample_time.to_string() +
    "ULL)%" +
    channel_count.to_string() +
    "ULL;const float x=upsampled[flat];const float a=expf(alpha[channel]);const float b=expf(beta[channel]);const float s=sinf(__fmul_rn(a,x));activated[flat]=__fadd_rn(x,__fdiv_rn(__fmul_rn(s,s),__fadd_rn(b,1.0e-9f)));}\nextern \"C\" __global__ void " +
    downsample_symbol +
    "(const float *activated,const float *filter,float *output){const unsigned long long flat=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(flat>=" +
    input_elements.to_string() +
    "ULL)return;const int t=(int)(flat%" +
    time_count.to_string() +
    "ULL);const unsigned long long outer=flat/" +
    time_count.to_string() +
    "ULL;float sum=0.0f;for(int k=0;k<12;k+=1){int source_t=t*2+k-5;if(source_t<0)source_t=0;if(source_t>=" +
    upsample_time.to_string() +
    ")source_t=" +
    (upsample_time - 1).to_string() +
    ";sum=__fadd_rn(sum,__fmul_rn(activated[outer*" +
    upsample_time.to_string() +
    "ULL+(unsigned long long)source_t],filter[k]));}output[flat]=sum;}\n",
  )
  {
    numerical_contract: OrderedF32UpsampleSnakeBetaDownsampleV1,
    batch_count,
    channel_count,
    time_count,
    ratio,
    kernel_size,
    upsample_time,
    input_element_count: input_elements,
    intermediate_element_count: intermediate_elements,
    output_element_count: input_elements,
    upsample_symbol,
    activation_symbol,
    downsample_symbol,
    source,
  }
}
