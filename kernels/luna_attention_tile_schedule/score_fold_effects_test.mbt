///|
/// Cross-owner visibility model: pending writes are unavailable until the
/// workgroup publishes them. The next fold consumes the previous output.
priv struct ScoreFoldModel {
  mut pending_scale : Bool
  mut pending_probability : Bool
  mut pending_output : Bool
  mut scales : Int
  mut probabilities : Int
  mut outputs : Int
}

///|
fn ScoreFoldModel::step(
  self : ScoreFoldModel,
  step : @luna_attention_tile_schedule.AttentionScoreFoldStep,
  tile : Int,
) -> Unit raise {
  match step {
    UpdateResidentStatisticsAndProbabilities => {
      assert_true(!self.pending_probability && !self.pending_output)
      self.pending_scale = true
      self.pending_probability = true
    }
    UpdateSharedMaximumAndScale => {
      assert_true(!self.pending_scale && !self.pending_output)
      self.pending_scale = true
    }
    RescaleSharedOutput => {
      assert_eq(self.scales, tile + 1)
      assert_true(!self.pending_scale && !self.pending_output)
      self.pending_output = true
    }
    UpdateSharedProbabilitiesAndDenominator => {
      assert_eq(self.scales, tile + 1)
      assert_true(!self.pending_scale && !self.pending_probability)
      self.pending_probability = true
    }
    PublishScoreFold => {
      if self.pending_scale {
        self.scales += 1
      }
      if self.pending_probability {
        self.probabilities += 1
      }
      if self.pending_output {
        self.outputs += 1
      }
      self.pending_scale = false
      self.pending_probability = false
      self.pending_output = false
    }
  }
}

///|
fn check_score_fold(
  steps : ReadOnlyArray[@luna_attention_tile_schedule.AttentionScoreFoldStep],
  tiles : Int,
  resident_output : Bool,
) -> Unit raise {
  let model : ScoreFoldModel = {
    pending_scale: false,
    pending_probability: false,
    pending_output: false,
    scales: 0,
    probabilities: 0,
    outputs: 0,
  }
  for tile in 0..<tiles {
    for step in steps {
      model.step(step, tile)
    }
    // PV may now consume probabilities and the rescaled accumulator.
    assert_true(
      !model.pending_scale &&
      !model.pending_probability &&
      !model.pending_output,
    )
    assert_eq(model.probabilities, tile + 1)
    assert_eq(model.outputs, if resident_output { 0 } else { tile + 1 })
  }
}

///|
test "score fold publishes shared data before cross-owner use" {
  for resident_statistics in [false, true] {
    for resident_output in [false, true] {
      let steps = @luna_attention_tile_schedule.plan_attention_score_fold(
        resident_statistics~,
        resident_output~,
      )
      for tiles in 0..<=9 {
        check_score_fold(steps, tiles, resident_output)
      }
      // The first and final publications have required cross-owner consumers.
      // Intermediate fences are preserved, not claimed minimal by this model.
      for index in [1, steps.length() - 1] {
        let kept = []
        for i, item in steps {
          if i != index {
            kept.push(item)
          }
        }
        let omitted = ReadOnlyArray::from_array(kept)
        try check_score_fold(omitted, 2, resident_output) catch {
          _ => ()
        } noraise {
          _ => fail("missing publication accepted")
        }
      }
    }
  }
}
