///|
pub(all) enum Fp8LaunchOwnerState {
  Available
  Acquired
  InUse
  Closed
} derive(Debug, Eq)

///|
priv enum Fp8LaunchAuthorityState {
  AuthorityAvailable
  AuthorityAcquired(UInt64)
  AuthorityInUse(UInt64)
  AuthorityClosed
}

///|
priv struct Fp8LaunchAuthority {
  digest : Fp8LaunchAbiDigest
  generation : UInt64
  mut next_epoch : UInt64
  mut last_released_epoch : UInt64
  mut state : Fp8LaunchAuthorityState
}

///|
pub struct Fp8DynamicLaunchOwner {
  priv abi : Fp8DynamicLaunchAbi
  priv authority : Fp8LaunchAuthority
}

///|
pub struct Fp8DynamicLaunchLease {
  priv authority : Fp8LaunchAuthority
  priv digest : Fp8LaunchAbiDigest
  priv generation : UInt64
  priv epoch : UInt64
}

///|
pub fn Fp8DynamicLaunchOwner::new(
  abi : Fp8DynamicLaunchAbi,
  generation~ : UInt64,
) -> Fp8DynamicLaunchOwner raise Fp8LaunchAbiError {
  if generation == 0UL {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  {
    abi,
    authority: {
      digest: abi.digest(),
      generation,
      next_epoch: 1UL,
      last_released_epoch: 0UL,
      state: AuthorityAvailable,
    },
  }
}

///|
pub fn Fp8DynamicLaunchOwner::abi(
  self : Fp8DynamicLaunchOwner,
) -> Fp8DynamicLaunchAbi {
  self.abi
}

///|
fn validate_lease_binding(
  authority : Fp8LaunchAuthority,
  digest : Fp8LaunchAbiDigest,
  generation : UInt64,
) -> Unit raise Fp8LaunchAbiError {
  if authority.digest != digest || authority.generation != generation {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
}

///|
pub fn Fp8DynamicLaunchOwner::state(
  self : Fp8DynamicLaunchOwner,
) -> Fp8LaunchOwnerState {
  match self.authority.state {
    AuthorityAvailable => Available
    AuthorityAcquired(_) => Acquired
    AuthorityInUse(_) => InUse
    AuthorityClosed => Closed
  }
}

///|
pub fn Fp8DynamicLaunchOwner::acquire(
  self : Fp8DynamicLaunchOwner,
  expected : Fp8DynamicLaunchAbi,
  generation~ : UInt64,
) -> Fp8DynamicLaunchLease raise Fp8LaunchAbiError {
  validate_lease_binding(self.authority, expected.digest(), generation)
  match self.authority.state {
    AuthorityAvailable => ()
    AuthorityAcquired(_) | AuthorityInUse(_) | AuthorityClosed =>
      raise InvalidFp8LaunchAbi(InvalidTransition)
  }
  if self.authority.next_epoch == 18446744073709551615UL {
    raise InvalidFp8LaunchAbi(EpochExhausted)
  }
  let epoch = self.authority.next_epoch
  self.authority.next_epoch += 1UL
  self.authority.state = AuthorityAcquired(epoch)
  { authority: self.authority, digest: expected.digest(), generation, epoch }
}

///|
pub fn Fp8DynamicLaunchLease::begin_use(
  self : Fp8DynamicLaunchLease,
  expected : Fp8DynamicLaunchAbi,
  generation~ : UInt64,
) -> Unit raise Fp8LaunchAbiError {
  validate_lease_binding(self.authority, expected.digest(), generation)
  if self.digest != expected.digest() || self.generation != generation {
    raise InvalidFp8LaunchAbi(BindingMismatch)
  }
  match self.authority.state {
    AuthorityAcquired(epoch) if epoch == self.epoch =>
      self.authority.state = AuthorityInUse(epoch)
    AuthorityClosed => raise InvalidFp8LaunchAbi(InvalidTransition)
    _ => raise InvalidFp8LaunchAbi(StaleLease)
  }
}

///|
pub fn Fp8DynamicLaunchLease::finish_use(
  self : Fp8DynamicLaunchLease,
) -> Unit raise Fp8LaunchAbiError {
  match self.authority.state {
    AuthorityInUse(epoch) if epoch == self.epoch =>
      self.authority.state = AuthorityAcquired(epoch)
    AuthorityClosed => raise InvalidFp8LaunchAbi(InvalidTransition)
    _ => raise InvalidFp8LaunchAbi(StaleLease)
  }
}

///|
pub fn Fp8DynamicLaunchLease::release(
  self : Fp8DynamicLaunchLease,
) -> Unit raise Fp8LaunchAbiError {
  match self.authority.state {
    AuthorityAcquired(epoch) if epoch == self.epoch => {
      self.authority.last_released_epoch = epoch
      self.authority.state = AuthorityAvailable
    }
    AuthorityAvailable if self.authority.last_released_epoch == self.epoch => ()
    AuthorityClosed => ()
    AuthorityInUse(epoch) if epoch == self.epoch =>
      raise InvalidFp8LaunchAbi(InvalidTransition)
    _ => raise InvalidFp8LaunchAbi(StaleLease)
  }
}

///|
/// Idempotently invalidates every outstanding lease. This pure host owner has
/// no CUDA handle; terminal cleanup consists of deterministic authority loss.
pub fn Fp8DynamicLaunchOwner::close(self : Fp8DynamicLaunchOwner) -> Unit {
  self.authority.state = AuthorityClosed
  self.authority.last_released_epoch = 0UL
}
