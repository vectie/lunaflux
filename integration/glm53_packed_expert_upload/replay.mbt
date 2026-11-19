///|
pub fn GlmDecoderSlice::replay_graph_count(self : Self) -> Int32 {
  match self.collective_queue {
    Some(queue) => queue.replay_graph_count()
    None => 0
  }
}

///|
pub fn GlmDecoderStage::replay_graph_count(self : Self) -> Int32 {
  self.decoder.replay_graph_count()
}
