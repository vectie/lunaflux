///|
pub fn RecurrentDeltaCudaSource::workspace_bytes(self : Self) -> Int64 {
  match self.chunk {
    None => 0L
    Some(plan) => plan.workspace_bytes()
  }
}

///|
pub fn RecurrentDeltaCudaSource::stage_count(self : Self) -> Int {
  match self.chunk {
    None => 1
    Some(plan) => plan.stage_count()
  }
}

///|
pub fn RecurrentDeltaCudaSource::stage_symbol(
  self : Self,
  stage : Int,
) -> String {
  if self.chunk is None {
    self.symbol
  } else {
    self.symbol +
    (match stage {
      0 => "_prepare"
      1 => "_solve"
      _ => "_publish"
    })
  }
}

///|
pub fn RecurrentDeltaCudaSource::stage_launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int, Int) {
  let (x, y, block) = self.launch()
  match self.chunk {
    None => (x, y, 1, block)
    Some(plan) => {
      let (rows, _, _) = self.plan.capacity()
      let (_, keys, values) = self.plan.geometry()
      match stage {
        0 => (x, y, 1, 128)
        1 => (x, y, (values + plan.value_tile() - 1) / plan.value_tile(), 32)
        _ =>
          (
            x,
            y,
            ((if rows > keys { rows } else { keys }) * values + 127) / 128,
            128,
          )
      }
    }
  }
}
