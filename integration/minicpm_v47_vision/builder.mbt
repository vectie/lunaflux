///|
priv struct Builder {
  source : StringBuilder
  entries : Array[Entry]
  stages : Array[Stage]
}

///|
fn Builder::new() -> Self {
  { source: StringBuilder(), entries: [], stages: [], }
}

///|
fn Builder::add(
  self : Self,
  name : String,
  launch : (Int32, Int32, Int32),
  parameters : Int32,
  bytes : Bytes,
  shared~ : Int32 = 0,
) -> Int32 raise {
  let index = self.entries.length()
  self.entries.push({ symbol: name, launch, parameters, shared_bytes: shared, })
  self.source.write_string(@utf8.decode(bytes))
  self.source.write_string("\n")
  index
}

///|
fn Builder::stage(
  self : Self,
  entry : Int32,
  arguments : Array[Argument],
) -> Unit raise {
  guard arguments.length() == self.entries[entry].parameters else {
    fail("vision stage argument count")
  }
  self.stages.push({ entry, arguments: ReadOnlyArray::from_array(arguments), })
}

///|
fn Builder::norm(
  self : Self,
  name : String,
  rows : Int32,
  width : Int32,
) -> Int32 raise {
  self.add(
    name,
    (rows, 1, 1),
    5,
    @common.norm_source(rows~, width~, symbol=name, counted=true),
  )
}

///|
fn Builder::linear(
  self : Self,
  name : String,
  rows : Int32,
  input : Int32,
  output : Int32,
  activation : Int32,
) -> Int32 raise {
  self.add(
    name,
    ((rows * output + 255) / 256, 1, 256),
    5,
    @common.linear_source(
      rows~,
      input_width=input,
      output_width=output,
      activation~,
      symbol=name,
      counted=true,
    ),
  )
}

///|
fn Builder::residual(self : Self, name : String, rows : Int32) -> Int32 raise {
  self.add(
    name,
    ((rows * 1152 + 255) / 256, 1, 256),
    4,
    @common.residual_source(rows~, width=1152, symbol=name, counted=true),
  )
}

///|
fn Builder::attention(
  self : Self,
  name : String,
  rows : Int32,
  segment : Int32,
) -> Int32 raise {
  let plan = @precision.SegmentedAttentionPrecision::new(
    maximum_rows=rows,
    maximum_segment=segment,
    head_count=16,
    head_width=72,
  )
  let source = @segmented.attention_source(plan, symbol=name)
  self.add(
    name,
    source.launch(),
    source.parameter_count(),
    source.source_bytes(),
    shared=source.shared_bytes(),
  )
}

///|
fn Builder::norm_stage(
  self : Self,
  entry : Int32,
  count : Plane,
  input : Plane,
  prefix : String,
  output : Plane,
) -> Unit raise {
  self.stage(entry, [
    Buffer(count),
    Buffer(input),
    Weight(prefix + ".weight"),
    Weight(prefix + ".bias"),
    Buffer(output),
  ])
}

///|
fn Builder::linear_stage(
  self : Self,
  entry : Int32,
  count : Plane,
  input : Plane,
  prefix : String,
  output : Plane,
) -> Unit raise {
  self.norm_stage(entry, count, input, prefix, output)
}

///|
fn Builder::attention_stage(
  self : Self,
  entry : Int32,
  count : Plane,
  starts : Plane,
  ends : Plane,
) -> Unit raise {
  self.stage(entry, [
    Buffer(count),
    Buffer(starts),
    Buffer(ends),
    Buffer(Query),
    Buffer(Key),
    Buffer(Value),
    Buffer(Attention),
  ])
}

///|
priv struct BlockEntries {
  norm : Int32
  square : Int32
  attention : Int32
  up : Int32
  down : Int32
  residual : Int32
}

///|
fn Builder::block_entries(
  self : Self,
  tag : String,
  rows : Int32,
  segment : Int32,
) -> BlockEntries raise {
  {
    norm: self.norm(tag + "_norm", rows, 1152),
    square: self.linear(tag + "_square", rows, 1152, 1152, 0),
    attention: self.attention(tag + "_attention", rows, segment),
    up: self.linear(tag + "_up", rows, 1152, 4304, 1),
    down: self.linear(tag + "_down", rows, 4304, 1152, 0),
    residual: self.residual(tag + "_residual", rows),
  }
}

///|
fn Builder::block(
  self : Self,
  entries : BlockEntries,
  count : Plane,
  starts : Plane,
  ends : Plane,
  layer : Int32,
) -> Unit raise {
  let prefix = "model.vision_tower.encoder.layers.\{layer}"
  self.norm_stage(
    entries.norm,
    count,
    State,
    prefix + ".layer_norm1",
    Normalized,
  )
  for (name, output) in [("q_proj", Query), ("k_proj", Key), ("v_proj", Value)] {
    self.linear_stage(
      entries.square,
      count,
      Normalized,
      prefix + ".self_attn." + name,
      output,
    )
  }
  self.attention_stage(entries.attention, count, starts, ends)
  self.linear_stage(
    entries.square,
    count,
    Attention,
    prefix + ".self_attn.out_proj",
    Branch,
  )
  self.stage(entries.residual, [
    Buffer(count),
    Buffer(State),
    Buffer(Branch),
    Buffer(State),
  ])
  self.norm_stage(
    entries.norm,
    count,
    State,
    prefix + ".layer_norm2",
    Normalized,
  )
  self.linear_stage(
    entries.up,
    count,
    Normalized,
    prefix + ".mlp.fc1",
    Intermediate,
  )
  self.linear_stage(
    entries.down,
    count,
    Intermediate,
    prefix + ".mlp.fc2",
    Branch,
  )
  self.stage(entries.residual, [
    Buffer(count),
    Buffer(State),
    Buffer(Branch),
    Buffer(State),
  ])
}
