///|
/// Bulk producers and matrix consumers bind the same immutable XOR layout.
/// Quantized bytes/scales stay unchanged. Each chunk publishes once and retires
/// all readers before storage reuse; matrix folds retain ascending K order.
fn fp4_fragment_pipeline(
  stage : Int,
  inner : Int,
  columns : Int,
  selected : Int,
  input_scales : Int64,
  weight_scales : Int64,
  symbol : String,
  work_tile? : String = "blockIdx.x",
  paired_weight_scales? : Int64,
  tile_columns? : Int = 32,
  packed_input? : Bool = false,
  pipeline? : @physical_tile_ir.ProducerConsumerPipeline,
  buffered? : Bool = false,
  borrowed_ring? : String,
) -> String {
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=128,
    vector=16,
    rotations=8,
    row_group=1,
  ).unwrap()
  let storage = @physical_tile_ir.OperandStorage::new(
    layout,
    if paired_weight_scales is Some(_) {
      [16, tile_columns, tile_columns]
    } else {
      [16, tile_columns]
    },
    owners=128,
    stages=pipeline.map(p => p.geometry().0).unwrap_or(1),
  ).unwrap()
  let left = storage.region(0).unwrap().geometry().0
  let right = storage.region(1).unwrap().geometry().0
  let scales = storage.slot_elements()
  let count = if paired_weight_scales is Some(_) { 3 } else { 2 }
  let copies = []
  for operand in 0..<(count * 2) {
    let owner = operand % count
    let is_left = owner == 0
    let is_scale = operand >= count
    let width = if is_scale { 16 } else { 128 }
    let rows = if is_left { 16 } else { tile_columns }
    let base = if is_scale {
      scales + (if is_left { 0 } else { 256 + (owner - 1) * tile_columns * 16 })
    } else {
      storage.region(owner).unwrap().geometry().0
    }
    let stride = if is_scale { inner / 16 } else { inner / 2 }
    let offset = if is_scale { "base_k/16" } else { "base_k/2" }
    let origin = if is_left {
      if packed_input {
        "packed"
      } else {
        "input"
      }
    } else {
      "w" + (stage + owner - 1).to_string()
    }
    let extra = if !is_scale {
      0L
    } else if is_left {
      input_scales
    } else if owner == 1 {
      weight_scales
    } else {
      paired_weight_scales.unwrap()
    }
    let row = if is_left && packed_input {
      "int source_row=group*16+row;bool valid=job_ids[row]>=0&&local>=0;"
    } else if is_left {
      let index = if stage < 2 { "job/" + selected.to_string() } else { "job" }
      "int job=job_ids[row];int source_row=" +
      index +
      ";bool valid=job>=0&&local>=0;"
    } else {
      "int source_row=(" +
      work_tile +
      "%" +
      ((columns + tile_columns - 1) / tile_columns).to_string() +
      ")*" +
      tile_columns.to_string() +
      "+row;bool valid=local>=0&&source_row<" +
      columns.to_string() +
      ";"
    }
    let physical = if is_scale { "col" } else { "(col^((row%8)*16))" }
    if is_scale && stride % 16 != 0 {
      copies.push(
        (
          $|for(int v=threadIdx.x;v<\{rows*width};v+=128) {
          $|  int row=v/\{width},col=v%\{width};\{row}
          $|  int offset=\{offset}+col;
          $|  tile[\{base}+row*\{width}+col]=valid&&offset<\{stride}?\{origin}[\{extra}ULL+(unsigned long long)source_row*\{stride}+offset]:0;
          $|}
        ),
      )
      continue
    }
    copies.push(
      (
        $|for(int v=threadIdx.x;v<\{rows*width/16};v+=128) {
        $|  int row=v/\{width/16},col=v%\{width/16}*16;\{row}
        $|  int offset=\{offset}+col,remaining=\{stride}-offset;int bytes=valid&&remaining>0?(remaining<16?remaining:16):0;
        $|  const unsigned char *src=\{origin}+\{extra}ULL+(unsigned long long)(valid?source_row:0)*\{stride}+(bytes?offset:0);
        $|  unsigned dst=(unsigned)__cvta_generic_to_shared(tile+\{base}+row*\{width}+\{physical});
        $|  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst),"l"(src),"r"(bytes):"memory");
        $|}
      ),
    )
  }
  let accumulator_offset = if tile_columns == 32 { "" } else { "+fragment*4" }
  let second = match paired_weight_scales {
    None => ""
    Some(_) => {
      let second_right = storage.region(2).unwrap().geometry().0
      (
        $|const unsigned *bp1=(const unsigned *)(tile+\{second_right}+wr*128);
        $|unsigned sb1=*(const unsigned *)(tile+\{scales+256+tile_columns*16}+wr*16+kk/16);
        $|\{symbol}_mma(d\{stage+1}\{accumulator_offset},a0p[lo],a1p[lo],a0p[hi],a1p[hi],bp1[lo],bp1[hi],sa,sb1);
      )
    }
  }
  let left_fragment =
    $|int lo=((kk/2+t*4)^((g%8)*16))/4,hi=((kk/2+(t+4)*4)^((g%8)*16))/4;
    $|unsigned sa=*(const unsigned *)(tile+\{scales}+(g+(t&1)*8)*16+kk/16);
  let right_fragment =
    $|int wr=(\{if pipeline is Some(_) {"(threadIdx.x%128)"} else {"threadIdx.x"}}/32)*8+g\{if tile_columns==32 {""} else {"+fragment*32"}};
    $|const unsigned *bp=(const unsigned *)(tile+\{right}+wr*128);
  let consume =
    $|unsigned sb=*(const unsigned *)(tile+\{scales+256}+wr*16+kk/16);
    $|\{symbol}_mma(d\{stage}\{accumulator_offset},a0p[lo],a1p[lo],a0p[hi],a1p[hi],bp[lo],bp[hi],sa,sb);
    $|\{second}
  // Partial evaluation over one fragment keeps the established declaration
  // order; wider ownership shares the left fragment across independent rights.
  let fragments = if tile_columns == 32 {
    right_fragment + "\n" + left_fragment + "\n" + consume
  } else {
    (
      $|\{left_fragment}
      $|#pragma unroll
      $|for(int fragment=0;fragment<\{tile_columns/32};++fragment) {
      $|\{right_fragment}
      $|\{consume}
      $|}
    )
  }
  let consume_chunk =
    $|for(int kk=0;kk<256&&base_k+kk<\{inner};kk+=64) {
    $|  const unsigned *a0p=(const unsigned *)(tile+\{left}+g*128);
    $|  const unsigned *a1p=(const unsigned *)(tile+\{left}+(g+8)*128);
    $|  \{fragments}
    $|}
  if buffered {
    let slot_bytes = scales + 256 + (count - 1) * tile_columns * 16
    let declaration = match borrowed_ring {
      Some(name) => "unsigned char *ring=" + name + ";"
      None => "__shared__ __align__(16) unsigned char ring[\{slot_bytes*2}];"
    }
    return (
      $|{ // cooperative operand ring=2; direct indexed rows, ascending K64 folds
      $|\{declaration}
      $|{unsigned char *tile=ring;int base_k=0;\{copies.join("\n")}}
      $|asm volatile("cp.async.commit_group;cp.async.wait_group 0;" ::: "memory");
      $|__syncthreads();
      $|for(int epoch=0;epoch<\{(inner+255)/256};++epoch) {
      $|  int base_k=epoch*256;
      $|  if(base_k+256<\{inner}) {unsigned char *tile=ring+((epoch+1)%2)*\{slot_bytes};int base_k=(epoch+1)*256;\{copies.join("\n")}}
      $|  asm volatile("cp.async.commit_group;" ::: "memory");
      $|  const unsigned char *tile=ring+(epoch%2)*\{slot_bytes};
      $|  \{consume_chunk}
      $|  asm volatile("cp.async.wait_group 0;" ::: "memory");
      $|  __syncthreads(); // next producer completion and current reader retirement
      $|}
      $|}
    )
  }
  if pipeline is Some(p) {
    let (stages, producers, consumers) = p.geometry()
    let slot_bytes = scales + 256 + (count - 1) * tile_columns * 16
    let chunks = (inner + 255) / 256
    // Publication and retirement are distinct phase-counted memory barriers.
    // Each has only its publishing role as arrivals; the other role waits
    // without contributing an arrival to this or a later epoch.
    return (
      $|{ // separate producer/consumer ring; unchanged ascending K folds
      $|__shared__ __align__(16) unsigned char ring[\{slot_bytes*stages}];
      $|__shared__ __align__(8) unsigned long long ready[\{stages}],retired[\{stages}];
      $|if(threadIdx.x==0) for(int s=0;s<\{stages};++s) {
      $|  unsigned r=(unsigned)__cvta_generic_to_shared(ready+s),f=(unsigned)__cvta_generic_to_shared(retired+s);
      $|  asm volatile("mbarrier.init.shared.b64 [%0], \{producers};" :: "r"(r):"memory");
      $|  asm volatile("mbarrier.init.shared.b64 [%0], \{consumers};" :: "r"(f):"memory");
      $|}
      $|__syncthreads();
      $|if(threadIdx.x<\{producers}) {
      $|  for(int epoch=0;epoch<\{chunks};++epoch) {
      $|    int slot=epoch%\{stages},base_k=epoch*256;
      $|    unsigned char *tile=ring+slot*\{slot_bytes};
      $|    unsigned r=(unsigned)__cvta_generic_to_shared(ready+slot),f=(unsigned)__cvta_generic_to_shared(retired+slot);
      $|    if(epoch>=\{stages}) {unsigned done=0;do {
      $|      asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared.b64 p,[%1],%2;selp.u32 %0,1,0,p;}" : "=r"(done):"r"(f),"r"((epoch/\{stages}-1)&1):"memory");
      $|    } while(!done);}
      $|    \{copies.join("\n")}
      $|    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      $|    asm volatile("mbarrier.arrive.shared.b64 _,[%0];" :: "r"(r):"memory");
      $|  }
      $|} else {
      $|  for(int epoch=0;epoch<\{chunks};++epoch) {
      $|    int slot=epoch%\{stages},base_k=epoch*256;
      $|    const unsigned char *tile=ring+slot*\{slot_bytes};
      $|    unsigned r=(unsigned)__cvta_generic_to_shared(ready+slot),f=(unsigned)__cvta_generic_to_shared(retired+slot),done=0;
      $|    do {asm volatile("{.reg .pred p;mbarrier.try_wait.parity.shared.b64 p,[%1],%2;selp.u32 %0,1,0,p;}" : "=r"(done):"r"(r),"r"((epoch/\{stages})&1):"memory");} while(!done);
      $|    \{consume_chunk}
      $|    asm volatile("mbarrier.arrive.shared.b64 _,[%0];" :: "r"(f):"memory");
      $|  }
      $|}
      $|__syncthreads();
      $|if(threadIdx.x==0) for(int s=0;s<\{stages};++s) {
      $|  unsigned r=(unsigned)__cvta_generic_to_shared(ready+s),f=(unsigned)__cvta_generic_to_shared(retired+s);
      $|  asm volatile("mbarrier.inval.shared.b64 [%0];mbarrier.inval.shared.b64 [%1];" :: "r"(r),"r"(f):"memory");
      $|}
      $|__syncthreads();
      $|}
    )
  }
  (
    $|{ // \{layout.canonical()}; ordered K chunks, explicit producer/reader lifetimes
    $|__shared__ __align__(16) unsigned char tile[\{scales+256+(count-1)*tile_columns*16}];
    $|for(int base_k=0;base_k<\{inner};base_k+=256) {
    $|  \{copies.join("\n")}
    $|  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");__syncthreads();
    $|  \{consume_chunk}
    $|  __syncthreads();
    $|}
    $|}
  )
}
