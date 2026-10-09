///|
pub(all) suberror ExportError {
  InvalidPlan
  DuplicateModule
  InvalidModule
}

///|
/// One independent CUDA translation unit. Never concatenate modules: some
/// renderer ABIs intentionally reuse fixed symbols across distinct shapes.
pub struct SourceUnit {
  name : String
  source : Bytes
  symbols : ReadOnlyArray[String]
  geometry : ReadOnlyArray[(String, Int64)]
  sha256 : String
}

///|
pub struct SourceExport {
  model_digest : String
  units : ReadOnlyArray[SourceUnit]
  timestep_counts : ReadOnlyArray[Int]
}

///|
fn unit(
  name : String,
  source : Bytes,
  symbols : ReadOnlyArray[String],
  geometry : ArrayView[(String, Int64)],
) -> SourceUnit raise ExportError {
  guard name.length() > 0 && source.length() > 0 && symbols.length() > 0 else {
    raise InvalidModule
  }
  for character in name {
    guard character is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_') else {
      raise InvalidModule
    }
  }
  let seen : Map[String, Unit] = Map([])
  let body = @utf8.decode(source) catch { _ => raise InvalidModule }
  for symbol in symbols {
    guard !seen.contains(symbol) && body.contains("void " + symbol + "(") else {
      raise InvalidModule
    }
    seen[symbol] = ()
  }
  {
    name,
    source,
    symbols,
    geometry: ReadOnlyArray::from_array(geometry.to_owned()),
    sha256: @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  }
}

///|
/// Stable source inventory, not an artifact/admission receipt. Source hashes
/// do not assert a binary exists or that any physical execution succeeded.
pub fn SourceExport::manifest(self : Self) -> Bytes {
  let out = StringBuilder()
  out.write_string(
    "lunaflux-h3-offline-source-v1\nmodel_plan=" + self.model_digest + "\n",
  )
  for count in self.timestep_counts {
    out.write_string("timestep_count=\{count}\n")
  }
  for entry in self.units {
    out.write_string(
      "module=\{entry.name}\tbytes=\{entry.source.length()}\tsha256=\{entry.sha256}\n",
    )
    for symbol in entry.symbols {
      out.write_string("symbol=\{entry.name}\t\{symbol}\n")
    }
    for (key, value) in entry.geometry {
      out.write_string("shape=\{entry.name}\t\{key}\t\{value}\n")
    }
  }
  @utf8.encode(out.to_string())
}

///|
/// Offline-only publication into an already existing empty directory. Every
/// write is exclusive; failures preserve partial output and never overwrite it.
/// The manifest is written last, so incomplete exports have no final inventory.
pub async fn SourceExport::write(self : Self, directory : String) -> Unit {
  for entry in self.units {
    @fs.write_file(
      directory + "/" + entry.name + ".cu",
      entry.source,
      create_mode=CreateNew,
      permission=0o444,
      sync=Full,
    )
  }
  @fs.write_file(
    directory + "/SOURCES.v1",
    self.manifest(),
    create_mode=CreateNew,
    permission=0o444,
    sync=Full,
  )
}
