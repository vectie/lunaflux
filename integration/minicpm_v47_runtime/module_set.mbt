///|
/// Owns the seven decoder modules, shared by text and image execution owners.
/// Close borrowing queues and frames before closing this set.
pub struct ModuleSet {
  priv modules : Array[@device.Module]
  priv mut bindings : @text.Modules?
  priv mut state : SessionState
}

///|
pub fn ModuleSet::new() -> Self {
  { modules: [], bindings: None, state: Fresh, }
}

///|
pub fn ModuleSet::prepare(
  self : Self,
  context : @device.Context,
  images : ModuleImages,
) -> Unit raise {
  guard self.state is Fresh &&
    !images.envelope.is_empty() &&
    !images.recurrent.is_empty() &&
    !images.moe.is_empty() &&
    !images.attention_common.is_empty() &&
    !images.attention_prefill.is_empty() &&
    !images.attention_decode.is_empty() &&
    !images.text.is_empty() else {
    raise @device.DeviceError::InvalidArgument(ModuleLoad)
  }
  self.state = Failed
  let load = fn(bytes : Bytes) raise {
    let aot = context.load_module(bytes)
    self.modules.push(aot)
    aot
  }
  self.bindings = Some({
    decoder: {
      envelope: load(images.envelope),
      recurrent: load(images.recurrent),
      moe: load(images.moe),
      attention: {
        common: load(images.attention_common),
        prefill: load(images.attention_prefill),
        decode: load(images.attention_decode),
      },
    },
    text: load(images.text),
  })
  self.state = Prepared
}

///|
pub fn ModuleSet::bindings(self : Self) -> @text.Modules raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(ModuleLoad)
  }
  self.bindings.unwrap()
}

///|
pub fn ModuleSet::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.bindings = None
  while self.modules.last() is Some(aot) {
    aot.close()
    ignore(self.modules.pop())
  }
  self.state = Closed
}
