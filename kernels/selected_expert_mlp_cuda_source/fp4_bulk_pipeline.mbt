///|
/// Byte-counted row transfers are a terminal device realization of the same
/// finite publication/retirement ring. No tensor descriptor or host readback.
fn fp4_bulk_pipeline(
  stage : Int,
  inner : Int,
  columns : Int,
  tile_columns : Int,
  input_scales : Int64,
  weight_scales : Int64,
  symbol : String,
  pipeline : @physical_tile_ir.ProducerConsumerPipeline,
  fragment_slots : Int,
  paired_weight_scales : Int64?,
  tiled? : Bool = false,
) -> String {
  let (stages, producers, consumers) = pipeline.geometry()
  let paired = paired_weight_scales is Some(_)
  let operands = if paired { 3 } else { 2 }
  // Preserve contiguous 128-byte producer transactions while rotating each
  // consumer row by one vector. Both ends consume the same physical layout.
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=128,
    vector=16,
    rotations=if tiled { 8 } else { 1 },
    row_group=1,
    row_stride=if tiled { 128 } else { 144 },
  ).unwrap()
  let rows = [16]
  for _ in 1..<operands {
    rows.push(tile_columns)
  }
  let storage = @physical_tile_ir.OperandStorage::new(
    layout,
    rows[:],
    owners=producers,
    stages~,
  ).unwrap()
  let stride = layout.row_stride()
  let scales = storage.slot_elements()
  let bytes = scales + 256 + (operands - 1) * tile_columns * 16
  let copies = []
  for operand in 0..<operands {
    let left = operand == 0
    let rows = if left { 16 } else { tile_columns }
    let (base, _, _) = storage.region(operand).unwrap().geometry()
    let scale_base = scales +
      (if left { 0 } else { 256 + (operand - 1) * tile_columns * 16 })
    let origin = if left {
      "packed"
    } else {
      "w" + (stage + operand - 1).to_string()
    }
    let scale_offset = if left {
      input_scales
    } else if operand == 1 {
      weight_scales
    } else {
      paired_weight_scales.unwrap()
    }
    let row = if left {
      "int source_row=group*16+row;bool valid=job_ids[row]>=0&&local>=0;"
    } else {
      "int source_row=(work_tile%\{(columns+tile_columns-1)/tile_columns})*\{tile_columns}+row;bool valid=local>=0&&source_row<\{columns};"
    }
    if tiled {
      let payload_count = rows * 128
      let scale_count = rows * 16
      let group = if left {
        "group"
      } else {
        "(work_tile%\{columns/tile_columns})"
      }
      copies.push(
        (
          $|if(threadIdx.x==0) {
          $|  unsigned d=(unsigned)__cvta_generic_to_shared(tile+\{base});
          $|  unsigned sd=(unsigned)__cvta_generic_to_shared(tile+\{scale_base});
          $|  const unsigned char *src=\{origin}+((unsigned long long)\{group}*\{inner/256}+epoch)*\{payload_count};
          $|  const unsigned char *ss=\{origin}+\{scale_offset}ULL+((unsigned long long)\{group}*\{inner/256}+epoch)*\{scale_count};
          $|  if(local>=0) {
          $|    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;" :: "r"(r),"r"(\{payload_count+scale_count}):"memory");
          $|    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" :: "r"(d),"l"(src),"r"(\{payload_count}),"r"(r):"memory");
          $|    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" :: "r"(sd),"l"(ss),"r"(\{scale_count}),"r"(r):"memory");
          $|  }
          $|}
          $|if(local<0) {
          $|  for(int v=threadIdx.x*16;v<\{payload_count};v+=\{producers*16}) *(uint4 *)(tile+\{base}+v)=make_uint4(0,0,0,0);
          $|  for(int v=threadIdx.x*16;v<\{scale_count};v+=\{producers*16}) *(uint4 *)(tile+\{scale_base}+v)=make_uint4(0,0,0,0);
          $|}
        ),
      )
      continue
    }
    copies.push(
      (
        $|for(int row=threadIdx.x;row<\{rows};row+=\{producers}) {
        $|  \{row}
        $|  int remaining=\{inner/2}-base_k/2;
        $|  int transferred=valid?(remaining<128?remaining:128):0;
        $|  unsigned char *dst=tile+\{base}+row*\{stride};
        $|  if(transferred>0) {
        $|    const unsigned char *src=\{origin}+(unsigned long long)source_row*\{inner/2}+base_k/2;
        $|    unsigned d=(unsigned)__cvta_generic_to_shared(dst);
        $|    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;" :: "r"(r),"r"(transferred):"memory");
        $|    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" :: "r"(d),"l"(src),"r"(transferred),"r"(r):"memory");
        $|  }
        $|  for(int z=transferred;z<128;z+=16) *(uint4 *)(dst+z)=make_uint4(0,0,0,0);
        $|  unsigned char *sd=tile+\{scale_base}+row*16;
        $|  const unsigned char *ss=\{origin}+\{scale_offset}ULL+(unsigned long long)(valid?source_row:0)*\{inner/16};
        $|  if(valid&&base_k/16+16<=\{inner/16}&&\{inner/16}%16==0) *(uint4 *)sd=*(const uint4 *)(ss+base_k/16);
        $|  else for(int z=0;z<16;++z) sd[z]=valid&&base_k/16+z<\{inner/16}?ss[base_k/16+z]:0;
        $|}
      ),
    )
  }
  let reads = fp4_fragment_reads(
    stage,
    inner,
    tile_columns,
    0,
    16 * stride,
    scales,
    symbol,
    "(threadIdx.x-\{producers})",
    paired,
    (16 + tile_columns) * stride,
    fragment_slots,
    team_columns=consumers / 4,
    swizzled=tiled,
    row_stride=stride,
  )
  (
    $|{ // bulk rows; producers=\{producers}; consumers=\{consumers}; stages=\{stages}
    $|__shared__ __align__(16) unsigned char bulk_ring[\{bytes*stages}];
    $|__shared__ __align__(8) unsigned long long bulk_ready[\{stages}],bulk_retired[\{stages}];
    $|if(threadIdx.x==0) for(int s=0;s<\{stages};++s) {
    $|  unsigned r=(unsigned)__cvta_generic_to_shared(bulk_ready+s),f=(unsigned)__cvta_generic_to_shared(bulk_retired+s);
    $|  asm volatile("mbarrier.init.shared.b64 [%0],\{producers};mbarrier.init.shared.b64 [%1],\{consumers};" :: "r"(r),"r"(f):"memory");
    $|  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    $|}
    $|__syncthreads();
    $|if(threadIdx.x<\{producers}) {
    $|  for(int epoch=0;epoch<\{(inner+255)/256};++epoch) {
    $|    int slot=epoch%\{stages},base_k=epoch*256;
    $|    unsigned char *tile=bulk_ring+slot*\{bytes};
    $|    unsigned r=(unsigned)__cvta_generic_to_shared(bulk_ready+slot),f=(unsigned)__cvta_generic_to_shared(bulk_retired+slot);
    $|    if(epoch>=\{stages}) {unsigned done=0;do {
    $|      asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared.b64 p,[%1],%2;selp.u32 %0,1,0,p;}" : "=r"(done):"r"(f),"r"((epoch/\{stages}-1)&1):"memory");
    $|    } while(!done);}
    $|    \{copies.join("\n")}
    $|    asm volatile("mbarrier.arrive.shared.b64 _,[%0];" :: "r"(r):"memory");
    $|  }
    $|} else {
    $|  for(int epoch=0;epoch<\{(inner+255)/256};++epoch) {
    $|    int slot=epoch%\{stages},base_k=epoch*256;
    $|    const unsigned char *tile=bulk_ring+slot*\{bytes};
    $|    unsigned r=(unsigned)__cvta_generic_to_shared(bulk_ready+slot),f=(unsigned)__cvta_generic_to_shared(bulk_retired+slot),done=0;
    $|    do {asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared.b64 p,[%1],%2;selp.u32 %0,1,0,p;}" : "=r"(done):"r"(r),"r"((epoch/\{stages})&1):"memory");} while(!done);
    $|    \{reads}
    $|    asm volatile("mbarrier.arrive.shared.b64 _,[%0];" :: "r"(f):"memory");
    $|  }
    $|}
    $|__syncthreads();
    $|if(threadIdx.x==0) for(int s=0;s<\{stages};++s) {
    $|  unsigned r=(unsigned)__cvta_generic_to_shared(bulk_ready+s),f=(unsigned)__cvta_generic_to_shared(bulk_retired+s);
    $|  asm volatile("mbarrier.inval.shared.b64 [%0];mbarrier.inval.shared.b64 [%1];" :: "r"(r),"r"(f):"memory");
    $|}
    $|__syncthreads();
    $|}
  )
}
