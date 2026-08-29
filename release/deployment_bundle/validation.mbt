///|
fn is_ascii_bundle_value(value : StringView) -> Bool {
  for unit in value {
    if !(unit is ('!'..='~')) {
      return false
    }
  }
  true
}

///|
fn locator_component_is_invalid(
  value : StringView,
  start : Int,
  end : Int,
) -> Bool {
  let length = end - start
  length == 0 ||
  (length == 1 && value[start] == '.') ||
  (length == 2 && value[start] == '.' && value[start + 1] == '.')
}

///|
fn validate_relative_locator(
  value : StringView,
  area : DeploymentBundleArea,
  limits : DeploymentBundleLimits,
) -> Unit raise DeploymentBundleError {
  if value.length() == 0 ||
    value.length() > limits.maximum_locator_bytes() ||
    value[0] == '/' ||
    value[value.length() - 1] == '/' {
    raise InvalidField(area)
  }
  let mut component_start = 0
  for index, unit in value {
    if !(unit is ('a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' | '/')) {
      raise InvalidField(area)
    }
    if unit == '/' {
      if locator_component_is_invalid(value, component_start, index) {
        raise InvalidField(area)
      }
      component_start = index + 1
    }
  }
  if locator_component_is_invalid(value, component_start, value.length()) {
    raise InvalidField(area)
  }
}

///|
fn validate_base_image(
  value : StringView,
  limits : DeploymentBundleLimits,
) -> Unit raise DeploymentBundleError {
  if value.length() == 0 ||
    value.length() > limits.maximum_base_image_bytes() ||
    !is_ascii_bundle_value(value) {
    raise InvalidField(BaseImage)
  }
  let marker = "@sha256:"
  if value.length() <= marker.length() + 64 {
    raise InvalidField(BaseImage)
  }
  let mut marker_at = -1
  for index in 0..<=(value.length() - marker.length()) {
    if value[index:index + marker.length()].to_owned() == marker {
      if marker_at >= 0 {
        raise InvalidField(BaseImage)
      }
      marker_at = index
    }
  }
  if marker_at <= 0 || marker_at + marker.length() + 64 != value.length() {
    raise InvalidField(BaseImage)
  }
  let repository = value[0:marker_at]
  if repository.contains("@") || repository.contains("://") {
    raise InvalidField(BaseImage)
  }
  let mut final_component_start = 0
  for index, unit in repository {
    if !(unit
      is ('a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' | ':' | '/')) {
      raise InvalidField(BaseImage)
    }
    if unit == '/' {
      if locator_component_is_invalid(repository, final_component_start, index) {
        raise InvalidField(BaseImage)
      }
      final_component_start = index + 1
    }
  }
  if locator_component_is_invalid(
      repository,
      final_component_start,
      repository.length(),
    ) ||
    repository[final_component_start:].contains(":") {
    raise InvalidField(BaseImage)
  }
  if !is_lower_sha256(value[marker_at + marker.length():]) {
    raise InvalidField(BaseImage)
  }
}

///|
fn require_digest(
  value : StringView,
  area : DeploymentBundleArea,
) -> String raise DeploymentBundleError {
  if !is_lower_sha256(value) {
    raise InvalidField(area)
  }
  value.to_owned()
}
