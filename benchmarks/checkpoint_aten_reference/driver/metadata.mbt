///|
struct Model {
  root : String
  files : Map[String, String]
  headers : Map[String, (Int64, Json)]
  hidden : Int
  intermediate : Int
  heads : Int
  kv_heads : Int
  dim : Int
  vocabulary : Int
  epsilon : Double
  theta : Double
}

///|
fn field(json : Json, key : String) -> Json raise {
  guard json is Object(map) && map.get(key) is Some(value) else {
    fail("missing JSON field: " + key)
  }
  value
}

///|
fn integer(json : Json) -> Int64 raise {
  guard json is Number(n, ..) &&
    n >= 0.0 &&
    n < 9007199254740992.0 &&
    n == n.trunc() else {
    fail("expected exact nonnegative integer")
  }
  n.to_int64()
}

///|
fn extent(json : Json, key : String) -> Int raise {
  let n = integer(field(json, key))
  guard n > 0L && n <= 2147483647L else { fail("invalid extent: " + key) }
  n.to_int()
}

///|
fn scalar(json : Json, key : String) -> Double raise {
  guard field(json, key) is Number(n, ..) &&
    !n.is_nan() &&
    !n.is_inf() &&
    n > 0.0 else {
    fail("invalid scalar: " + key)
  }
  n
}

///|
fn Model::new(session : @native.Session, root : String) -> Self raise {
  let json = @json.parse(@utf8.decode(session.bytes(root + "/config.json")))
  guard field(json, "model_type") is String("qwen3_vl") else {
    fail(
      "reference implements original text-only Qwen3-VL encoder, not another model",
    )
  }
  let text = field(json, "text_config")
  guard field(text, "hidden_act") is String("silu") &&
    field(text, "attention_bias") is False else {
    fail("unsupported original encoder arithmetic")
  }
  let index = @json.parse(
    @utf8.decode(session.bytes(root + "/model.safetensors.index.json")),
  )
  guard field(index, "weight_map") is Object(map) else {
    fail("invalid weight index")
  }
  let files : Map[String, String] = Map([])
  for name, value in map {
    guard value is String(path) && !path.contains("/") && !path.contains("..") else {
      fail("invalid shard filename")
    }
    files[name] = path
  }
  let rotary = field(text, "rope_scaling")
  guard field(rotary, "rope_type") is String("default") else {
    fail("scaled/visual rotary is outside this text-only reference")
  }
  {
    root,
    files,
    headers: Map([]),
    hidden: extent(text, "hidden_size"),
    intermediate: extent(text, "intermediate_size"),
    heads: extent(text, "num_attention_heads"),
    kv_heads: extent(text, "num_key_value_heads"),
    dim: extent(text, "head_dim"),
    vocabulary: extent(text, "vocab_size"),
    epsilon: scalar(text, "rms_norm_eps"),
    theta: scalar(text, "rope_theta"),
  }
}

///|
fn Model::region(
  self : Self,
  session : @native.Session,
  name : String,
  rows : Int,
  columns : Int,
) -> (String, Int64) raise {
  guard self.files.get(name) is Some(shard) else {
    fail("missing original weight: " + name)
  }
  let path = self.root + "/" + shard
  let (start, json) = match self.headers.get(shard) {
    Some(header) => header
    None => {
      let prefix = session.bytes(path, length=8)
      let size = for i in 0..<8; n = 0UL {
        continue n | (prefix[i].to_uint64() << (i * 8))
      } nobreak {
        n
      }
      guard size > 0UL && size <= 2097152UL else {
        fail("header exceeds reference metadata budget")
      }
      let json = @json.parse(
        @utf8.decode(session.bytes(path, offset=8L, length=size.to_int())),
      )
      let header = (8L + size.reinterpret_as_int64(), json)
      self.headers[shard] = header
      header
    }
  }
  let entry = field(json, name)
  guard field(entry, "dtype") is String("BF16") &&
    field(entry, "shape") is Array(shape) &&
    field(entry, "data_offsets") is Array(offsets) &&
    offsets.length() == 2 else {
    fail("invalid BF16 weight entry: " + name)
  }
  let expected : Array[Int] = if rows == 1 && shape.length() == 1 {
    [columns]
  } else {
    [rows, columns]
  }
  guard shape.length() == expected.length() else {
    fail("weight rank mismatch: " + name)
  }
  for i, value in expected {
    guard integer(shape[i]) == value.to_int64() else {
      fail("weight shape mismatch: " + name)
    }
  }
  let offset = integer(offsets[0])
  let end = integer(offsets[1])
  guard end - offset == rows.to_int64() * columns.to_int64() * 2L else {
    fail("weight extent mismatch: " + name)
  }
  (path, start + offset)
}

///|
fn Model::weight(
  self : Self,
  session : @native.Session,
  name : String,
  rows : Int,
  columns : Int,
) -> @native.Tensor raise {
  let (path, offset) = self.region(session, name, rows, columns)
  session.load(path, offset, rows, columns)
}
