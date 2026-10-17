///|
/// Immutable ownership of a finite ring. Slot publication follows completion
/// of every producer; reuse follows retirement of every consumer. Epochs, not
/// backend thread/barrier IDs, define the dependency edges.
pub struct ProducerConsumerPipeline {
  priv stages : Int
  priv producers : Int
  priv consumers : Int
} derive(Eq, Debug)

///|
pub extend ProducerConsumerPipeline with Eq::{equal, not_equal}

///|
pub extend ProducerConsumerPipeline with @debug.Debug::{to_repr}

///|
pub fn ProducerConsumerPipeline::new(
  stages~ : Int,
  producers~ : Int,
  consumers~ : Int,
) -> Self? {
  guard stages >= 2 &&
    stages <= 8 &&
    producers > 0 &&
    consumers > 0 &&
    producers <= 1024 &&
    consumers <= 1024 else {
    return None
  }
  Some({ stages, producers, consumers, })
}

///|
pub fn ProducerConsumerPipeline::geometry(self : Self) -> (Int, Int, Int) {
  (self.stages, self.producers, self.consumers)
}

///|
pub fn ProducerConsumerPipeline::slot(self : Self, epoch : Int) -> Int? {
  if epoch < 0 {
    None
  } else {
    Some(epoch % self.stages)
  }
}

///|
pub fn ProducerConsumerPipeline::reuse_after(self : Self, epoch : Int) -> Int? {
  if epoch < self.stages {
    None
  } else {
    Some(epoch - self.stages)
  }
}
