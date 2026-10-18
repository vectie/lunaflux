///|
/// A selection consumer determines what information must survive the shard
/// boundary. Greedy candidates are insufficient for temperature/top-k/top-p.
/// FullLogits preserves every model logit and leaves sampling outside the model.
pub(all) enum VocabularyPublication {
  GreedyCandidates
  FullLogits
} derive(Debug, Eq)

///|
pub extend VocabularyPublication with @debug.Debug::{to_repr}

///|
pub extend VocabularyPublication with Eq::{equal, not_equal}

///|
pub fn VocabularyShard::publication(self : Self) -> VocabularyPublication {
  self.publication
}

///|
/// Collective slots are equal width even for an uneven vocabulary partition.
/// Padding is transport-only, never a token or an input to a sampler.
pub fn VocabularyShard::padded_columns(self : Self) -> Int {
  (self.vocabulary - 1) / self.world + 1
}

///|
pub fn VocabularyShard::transfer_width(self : Self) -> Int {
  match self.publication {
    GreedyCandidates => self.candidate_width()
    FullLogits => self.outputs * self.padded_columns()
  }
}

///|
pub fn VocabularyShard::dense_logits_bytes(self : Self) -> Int64 {
  match self.publication {
    GreedyCandidates => 0L
    FullLogits => self.outputs.to_int64() * self.vocabulary.to_int64() * 4L
  }
}
