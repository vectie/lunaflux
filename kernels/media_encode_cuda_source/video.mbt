///|
fn video_body(op : @plan.Operation) -> String raise @plan.PlanError {
  let (_, count) = op.signature()
  let guard_flat = "if(f>=\{count}ULL)return;\n"
  match op {
    Zeros(..) => guard_flat + "out[f]=0.0f;"
    VideoSlice(
      frames~,
      height~,
      width~,
      start_frame~,
      start_y~,
      start_x~,
      tile_frames~,
      tile_height~,
      tile_width~
    ) =>
      guard_flat +
      (
        $|const int x=f%\{tile_width}ULL,y=(f/\{tile_width}ULL)%\{tile_height}ULL,t=(f/(\{tile_width}ULL*\{tile_height}ULL))%\{tile_frames}ULL,ch=f/(\{tile_width}ULL*\{tile_height}ULL*\{tile_frames}ULL);
        $|const int ti=min(\{frames - 1},\{start_frame}+t);
        $|out[f]=a[((ch*\{frames}ULL+ti)*\{height}ULL+y+\{start_y})*\{width}ULL+x+\{start_x}];
      )
    Conv3d(
      input_channels~,
      output_channels=_,
      frames~,
      height~,
      width~,
      kernel~,
      temporal_stride~,
      spatial_stride~,
      spatial_before~,
      spatial_after~
    ) => {
      let (ot, oh, ow) = @plan.conv3d_shape(
        frames, height, width, kernel, temporal_stride, spatial_stride, spatial_before,
        spatial_after,
      )
      guard_flat +
      (
        $|const int xo=f%\{ow}ULL,yo=(f/\{ow}ULL)%\{oh}ULL,to=(f/(\{ow}ULL*\{oh}ULL))%\{ot}ULL,co=f/(\{ow}ULL*\{oh}ULL*\{ot}ULL);
        $|float sum=c[co];
        $|for(int ci=0;ci<\{input_channels};++ci)for(int kt=0;kt<\{kernel};++kt){
        $|const int ti=to*\{temporal_stride}+kt-\{kernel - 1};if(ti<0||ti>=\{frames})continue;
        $|for(int ky=0;ky<\{kernel};++ky){int yi=yo*\{spatial_stride}+ky-\{spatial_before};yi=yi<0?-yi:(yi>=\{height}?2*\{height}-2-yi:yi);
        $|for(int kx=0;kx<\{kernel};++kx){int xi=xo*\{spatial_stride}+kx-\{spatial_before};xi=xi<0?-xi:(xi>=\{width}?2*\{width}-2-xi:xi);
        $|const unsigned long long ix=((ci*\{frames}ULL+ti)*\{height}ULL+yi)*\{width}ULL+xi;
        $|const unsigned long long wx=((((co*\{input_channels}ULL+ci)*\{kernel}ULL+kt)*\{kernel}ULL+ky)*\{kernel}ULL+kx);
        $|sum=__fadd_rn(sum,__fmul_rn(a[ix],b[wx]));}}}out[f]=sum;
      )
    }
    GroupNormSilu(channels~, frames~, height~, width~, groups~) =>
      (
        $|if(f>=\{frames * groups}ULL)return;const int t=f%\{frames}ULL,g=f/\{frames}ULL;
        $|const unsigned long long spatial=\{height}ULL*\{width}ULL;float mean=0.0f;
        $|for(int j=0;j<\{channels / groups};++j)for(unsigned long long p=0;p<spatial;++p)mean=__fadd_rn(mean,a[((g*\{channels / groups}ULL+j)*\{frames}ULL+t)*spatial+p]);
        $|mean=__fdiv_rn(mean,(float)(\{channels / groups}ULL*spatial));float variance=0.0f;
        $|for(int j=0;j<\{channels / groups};++j)for(unsigned long long p=0;p<spatial;++p){float v=__fsub_rn(a[((g*\{channels / groups}ULL+j)*\{frames}ULL+t)*spatial+p],mean);variance=__fadd_rn(variance,__fmul_rn(v,v));}
        $|float inv=rsqrtf(__fadd_rn(__fdiv_rn(variance,(float)(\{channels / groups}ULL*spatial)),1e-6f));
        $|for(int j=0;j<\{channels / groups};++j){const int ch=g*\{channels / groups}+j;for(unsigned long long p=0;p<spatial;++p){const unsigned long long ix=(ch*\{frames}ULL+t)*spatial+p;float v=__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(a[ix],mean),inv),b[ch]),c[ch]);out[ix]=__fdiv_rn(v,__fadd_rn(1.0f,expf(-v)));}}
      )
    InsertTile(
      channels=_,
      frames=_,
      height~,
      width~,
      tile_height~,
      tile_width~,
      y~,
      x~,
      copy_height~,
      copy_width~,
      overlap_y~,
      overlap_x~
    ) =>
      guard_flat +
      (
        $|const int xo=f%\{width}ULL,yo=(f/\{width}ULL)%\{height}ULL;const unsigned long long ct=f/(\{width}ULL*\{height}ULL);
        $|if(xo<\{x}||xo>=\{x + copy_width}||yo<\{y}||yo>=\{y + copy_height}){out[f]=a[f];return;}
        $|const int px=xo-\{x},py=yo-\{y};const unsigned long long base=ct*\{tile_height}ULL*\{tile_width}ULL;
        $|float value=b[base+py*\{tile_width}ULL+px];
      ) +
      (if overlap_y > 0 {
        "if(py<\{overlap_y}){float w=__fdiv_rn((float)py,\{overlap_y}.0f);value=__fadd_rn(__fmul_rn(c[base+(\{tile_height - overlap_y}+py)*\{tile_width}ULL+px],__fsub_rn(1.0f,w)),__fmul_rn(value,w));}\n"
      } else {
        ""
      }) +
      (if overlap_x > 0 {
        "if(px<\{overlap_x}){float w=__fdiv_rn((float)px,\{overlap_x}.0f);value=__fadd_rn(__fmul_rn(d[base+py*\{tile_width}ULL+\{tile_width - overlap_x}+px],__fsub_rn(1.0f,w)),__fmul_rn(value,w));}\n"
      } else {
        ""
      }) +
      "out[f]=value;"
    InsertTime(
      channels=_,
      frames~,
      height~,
      width~,
      tile_frames~,
      start~,
      count~
    ) =>
      guard_flat +
      (
        $|const unsigned long long spatial=\{height}ULL*\{width}ULL,p=f%spatial;const int t=(f/spatial)%\{frames}ULL,ch=f/(spatial*\{frames}ULL);
        $|out[f]=(t>=\{start}&&t<\{start + count})?b[(ch*\{tile_frames}ULL+t-\{start})*spatial+p]:a[f];
      )
    GaussianPatch(channels~, frames~, height~, width~) =>
      guard_flat +
      (
        $|const int dx=f%2ULL,dy=(f/2ULL)%2ULL,ch=(f/4ULL)%\{channels}ULL;
        $|const unsigned long long row=f/(\{channels}ULL*4ULL);const int x=(row%\{width / 2}ULL)*2+dx,y=((row/\{width / 2}ULL)%\{height / 2}ULL)*2+dy,t=row/(\{width / 2}ULL*\{height / 2}ULL);
        $|const unsigned long long ix=((ch*\{frames}ULL+t)*\{height}ULL+y)*\{width}ULL+x;
        $|const float mean=a[ix],logvar=fminf(20.0f,fmaxf(-30.0f,a[ix+\{channels}ULL*\{frames}ULL*\{height}ULL*\{width}ULL]));
        $|const float sampled=__fadd_rn(__fmul_rn(b[ix],expf(__fmul_rn(0.5f,logvar))),mean);
        $|const float latent=__half2float(__float2half_rn(sampled));out[f]=__fdiv_rn(__fsub_rn(latent,c[ch]),d[ch]);
      )
    _ => raise @plan.PlanError::InvalidGeometry
  }
}
