///|
/// Lower immutable read/consume actions into register lifetimes. A published
/// shared epoch remains read-only until every action retires. This does not
/// move a matrix operation across another matrix operation in the same fold.
fn fp4_fragment_reads(
  stage : Int,
  inner : Int,
  columns : Int,
  left : Int,
  right : Int,
  scales : Int,
  symbol : String,
  consumer_owner : String,
  paired : Bool,
  second_right : Int,
  slots : Int,
) -> String {
  let plan = @physical_tile_ir.OrderedOperandReads::new(items=4, slots~).unwrap()
  let declarations = []
  for slot in 0..<slots {
    for name in ["a0", "a1", "a2", "a3", "sa"] {
      declarations.push("unsigned r\{slot}_\{name}=0;")
    }
    for fragment in 0..<(columns / 32) {
      for operand in 0..<(if paired { 2 } else { 1 }) {
        for word in ["b0", "b1", "sb"] {
          declarations.push(
            "unsigned r\{slot}_f\{fragment}_o\{operand}_\{word}=0;",
          )
        }
      }
    }
  }
  let actions = []
  for action in plan.actions() {
    match action {
      ReadOperand(item~, slot~) => {
        let kk = item * 64
        let reads = [
          "const unsigned *a0p=(const unsigned *)(tile+\{left}+g*128);",
          "const unsigned *a1p=(const unsigned *)(tile+\{left}+(g+8)*128);",
          "int lo=((\{kk/2}+t*4)^((g%8)*16))/4,hi=((\{kk/2}+(t+4)*4)^((g%8)*16))/4;",
          "r\{slot}_a0=a0p[lo];r\{slot}_a1=a1p[lo];r\{slot}_a2=a0p[hi];r\{slot}_a3=a1p[hi];",
          "r\{slot}_sa=*(const unsigned *)(tile+\{scales}+(g+(t&1)*8)*16+\{kk/16});",
        ]
        for fragment in 0..<(columns / 32) {
          for operand in 0..<(if paired { 2 } else { 1 }) {
            let base = if operand == 0 { right } else { second_right }
            let scale = scales + 256 + operand * columns * 16
            let prefix = "r\{slot}_f\{fragment}_o\{operand}_"
            reads.push(
              "{int wr=(\{consumer_owner}/32)*8+g+\{fragment*32};const unsigned *bp=(const unsigned *)(tile+\{base}+wr*128);\{prefix}b0=bp[lo];\{prefix}b1=bp[hi];\{prefix}sb=*(const unsigned *)(tile+\{scale}+wr*16+\{kk/16});}",
            )
          }
        }
        actions.push("if(base_k+\{kk}<\{inner}) {\n\{reads.join("\n")}\n}")
      }
      ConsumeOperand(item~, slot~) => {
        let calls = []
        for fragment in 0..<(columns / 32) {
          for operand in 0..<(if paired { 2 } else { 1 }) {
            let prefix = "r\{slot}_f\{fragment}_o\{operand}_"
            calls.push(
              "\{symbol}_mma(d\{stage+operand}+\{fragment*4},r\{slot}_a0,r\{slot}_a1,r\{slot}_a2,r\{slot}_a3,\{prefix}b0,\{prefix}b1,r\{slot}_sa,\{prefix}sb);",
            )
          }
        }
        actions.push("if(base_k+\{item*64}<\{inner}) {\n\{calls.join("\n")}\n}")
      }
    }
  }
  "{ // \{plan.canonical()}\n\{declarations.join("\n")}\n\{actions.join("\n")}\n}"
}
