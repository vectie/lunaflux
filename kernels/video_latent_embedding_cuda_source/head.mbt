///|
/// Final affine LayerNorm (not RMSNorm) and F32 output projection directly to
/// channel-major video. Patch axis order is [channel,pt,ph,pw].
pub fn decoder_head_source(
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  hidden~ : Int,
  patch_t~ : Int,
  patch_hw~ : Int,
  channels~ : Int,
  prefix~ : String,
) -> Bytes raise EmbeddingError {
  guard temporal > 0 &&
    height > 0 &&
    width > 0 &&
    patch_t > 0 &&
    patch_hw > 0 &&
    channels > 0 &&
    temporal.to_int64() * height.to_int64() <= 2147483647L / width.to_int64() else {
    raise InvalidGeometry
  }
  let positions = temporal * height * width
  let checked = render(
    batch=1,
    positions~,
    input_channels=hidden,
    hidden_channels=hidden,
    register_rows=4,
    symbol=prefix + "_head_norm",
  )
  let patch64 = channels.to_int64() * patch_t.to_int64()
  guard patch64 <=
    2147483647L /
    patch_hw.to_int64() /
    patch_hw.to_int64() /
    positions.to_int64() else {
    raise InvalidGeometry
  }
  let patch = channels * patch_t * patch_hw * patch_hw
  let rows = positions + 5
  let _ = checked
  @utf8.encode(
    (
      $|#include <math.h>
      $|extern "C" __global__ void \{prefix}_head_norm(const float *input,const float *gamma,const float *bias,float *output){
      $| const unsigned row=blockIdx.x;if(row>=\{rows}U||threadIdx.x!=0)return;float mean=0.0f;
      $| for(int c=0;c<\{hidden};++c)mean=__fadd_rn(mean,input[(unsigned long long)row*\{hidden}ULL+c]);mean=__fdiv_rn(mean,\{hidden}.0f);float variance=0.0f;
      $| for(int c=0;c<\{hidden};++c){float x=__fsub_rn(input[(unsigned long long)row*\{hidden}ULL+c],mean);variance=__fadd_rn(variance,__fmul_rn(x,x));}float scale=rsqrtf(__fadd_rn(__fdiv_rn(variance,\{hidden}.0f),1.0e-5f));
      $| for(int c=0;c<\{hidden};++c)output[(unsigned long long)row*\{hidden}ULL+c]=__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(input[(unsigned long long)row*\{hidden}ULL+c],mean),scale),gamma[c]),bias[c]);
      $|}
      $|extern "C" __global__ void \{prefix}_head_unpack(const float *input,const float *weight,const float *bias,float *output){
      $| const unsigned long long i=(unsigned long long)blockIdx.x*256ULL+threadIdx.x;if(i>=\{positions.to_int64()*patch.to_int64()}ULL)return;
      $| const unsigned long long row=i/\{patch}ULL,p=i%\{patch}ULL;float sum=0.0f;
      $| for(int k=0;k<\{hidden};++k)sum=__fadd_rn(sum,__fmul_rn(input[row*\{hidden}ULL+k],weight[p*\{hidden}ULL+k]));sum=__fadd_rn(sum,bias[p]);
      $| const unsigned long long pw=p%\{patch_hw}ULL,ph=(p/\{patch_hw}ULL)%\{patch_hw}ULL,pt=(p/(\{patch_hw}ULL*\{patch_hw}ULL))%\{patch_t}ULL,c=p/(\{patch_hw}ULL*\{patch_hw}ULL*\{patch_t}ULL);
      $| const unsigned long long w=row%\{width}ULL,h=(row/\{width}ULL)%\{height}ULL,t=row/(\{height}ULL*\{width}ULL);
      $| output[((c*\{temporal*patch_t}ULL+t*\{patch_t}ULL+pt)*\{height*patch_hw}ULL+h*\{patch_hw}ULL+ph)*\{width*patch_hw}ULL+w*\{patch_hw}ULL+pw]=sum;
      $|}
    ),
  )
}
