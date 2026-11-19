///|
pub fn DeploymentBundleLimits::new(
  maximum_bytes~ : Int32,
  maximum_base_image_bytes~ : Int32,
  maximum_locator_bytes~ : Int32,
) -> DeploymentBundleLimits raise DeploymentBundleError {
  if maximum_bytes <= 0 ||
    maximum_bytes > 1048576 ||
    maximum_base_image_bytes <= 0 ||
    maximum_base_image_bytes > maximum_bytes ||
    maximum_locator_bytes <= 0 ||
    maximum_locator_bytes > maximum_bytes {
    raise InvalidLimit
  }
  { maximum_bytes, maximum_base_image_bytes, maximum_locator_bytes, }
}

///|
pub fn DeploymentBundleLimits::maximum_bytes(
  self : DeploymentBundleLimits,
) -> Int32 {
  self.maximum_bytes
}

///|
pub fn DeploymentBundleLimits::maximum_base_image_bytes(
  self : DeploymentBundleLimits,
) -> Int32 {
  self.maximum_base_image_bytes
}

///|
pub fn DeploymentBundleLimits::maximum_locator_bytes(
  self : DeploymentBundleLimits,
) -> Int32 {
  self.maximum_locator_bytes
}
