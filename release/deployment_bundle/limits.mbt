///|
pub fn DeploymentBundleLimits::new(
  maximum_bytes~ : Int,
  maximum_base_image_bytes~ : Int,
  maximum_locator_bytes~ : Int,
) -> DeploymentBundleLimits raise DeploymentBundleError {
  if maximum_bytes <= 0 ||
    maximum_bytes > 1048576 ||
    maximum_base_image_bytes <= 0 ||
    maximum_base_image_bytes > maximum_bytes ||
    maximum_locator_bytes <= 0 ||
    maximum_locator_bytes > maximum_bytes {
    raise InvalidLimit
  }
  { maximum_bytes, maximum_base_image_bytes, maximum_locator_bytes }
}

///|
pub fn DeploymentBundleLimits::maximum_bytes(
  self : DeploymentBundleLimits,
) -> Int {
  self.maximum_bytes
}

///|
pub fn DeploymentBundleLimits::maximum_base_image_bytes(
  self : DeploymentBundleLimits,
) -> Int {
  self.maximum_base_image_bytes
}

///|
pub fn DeploymentBundleLimits::maximum_locator_bytes(
  self : DeploymentBundleLimits,
) -> Int {
  self.maximum_locator_bytes
}
