///|
pub struct VisionSource {
  plan : @vision.VisionPlan
  symbols : ReadOnlyArray[String]
  source : Bytes
}

///|
fn geometry_source(plan : @vision.VisionPlan, prefix : String) -> String {
  let b = StringBuilder()
  b.write_string(
    "__device__ void \{prefix}_geometry(int row,int* start,int* end,int* y,int* x,int* h,int* w){\n",
  )
  let mut start = 0
  for grid in plan.grids {
    let area = grid.height * grid.width
    let end = start + grid.time * area
    b.write_string(
      "if(row<\{end}){int r=row-\{start};int p=r%\{area};int g=p/4;*start=\{start}+(r/\{area})*\{area};*end=*start+\{area};*h=\{grid.height};*w=\{grid.width};*y=(g/\{grid.width / 2})*2+p%4/2;*x=(g%\{grid.width / 2})*2+p%2;return;}\n",
    )
    start = end
  }
  b.write_string("*start=*end=*y=*x=0;*h=*w=2;}\n")
  b.to_string()
}

///|
pub fn build_source(
  plan : @vision.VisionPlan,
  prefix~ : String,
) -> VisionSource raise {
  let symbols : ReadOnlyArray[String] = [
    prefix + "_patch",
    prefix + "_position",
    prefix + "_norm",
    prefix + "_qkv",
    prefix + "_rotary",
    prefix + "_attention",
    prefix + "_out",
    prefix + "_residual",
    prefix + "_fc1",
    prefix + "_fc2",
    prefix + "_merged_norm",
    prefix + "_merge_fc1",
    prefix + "_merge_fc2",
  ]
  let r = plan.rows
  let m = plan.merged_rows
  let source = @cuda_source.linear_source(
      rows=r,
      input_width=1536,
      output_width=1152,
      activation=0,
      symbol=symbols[0],
    ) +
    @cuda_source.position_source(
      rows=r,
      geometry=geometry_source(plan, prefix),
      prefix~,
    ) +
    @cuda_source.norm_source(rows=r, width=1152, symbol=symbols[2]) +
    @cuda_source.linear_source(
      rows=r,
      input_width=1152,
      output_width=3456,
      activation=0,
      symbol=symbols[3],
    ) +
    @cuda_source.linear_source(
      rows=r,
      input_width=1152,
      output_width=1152,
      activation=0,
      symbol=symbols[6],
    ) +
    @cuda_source.residual_source(rows=r, width=1152, symbol=symbols[7]) +
    @cuda_source.linear_source(
      rows=r,
      input_width=1152,
      output_width=4304,
      activation=1,
      symbol=symbols[8],
    ) +
    @cuda_source.linear_source(
      rows=r,
      input_width=4304,
      output_width=1152,
      activation=0,
      symbol=symbols[9],
    ) +
    @cuda_source.norm_source(rows=m, width=4608, symbol=symbols[10]) +
    @cuda_source.linear_source(
      rows=m,
      input_width=4608,
      output_width=4608,
      activation=2,
      symbol=symbols[11],
    ) +
    @cuda_source.linear_source(
      rows=m,
      input_width=4608,
      output_width=5120,
      activation=0,
      symbol=symbols[12],
    )
  { plan, symbols, source, }
}
