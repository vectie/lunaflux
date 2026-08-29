///|
/// Startup-fixed direct index from one lower worker publication lane to its
/// owning outer session lane. The exact lower generation authenticates reuse.
priv struct LunaOnlinePublicationRoutes {
  session_lanes : FixedArray[Int]
  generations : FixedArray[UInt64]
  mut lookups : UInt64
  mut rejections : UInt64
  mut max_probes : Int
}

///|
fn route_counter_increment(value : UInt64) -> UInt64 {
  if value == 0xffffffffffffffffUL {
    value
  } else {
    value + 1UL
  }
}

///|
fn LunaOnlinePublicationRoutes::new(
  capacity : Int,
) -> LunaOnlinePublicationRoutes {
  if capacity <= 0 {
    abort("Luna online publication route capacity invariant")
  }
  {
    session_lanes: FixedArray::make(capacity, -1),
    generations: FixedArray::make(capacity, 0UL),
    lookups: 0UL,
    rejections: 0UL,
    max_probes: 0,
  }
}

///|
fn LunaOnlinePublicationRoutes::capacity(
  self : LunaOnlinePublicationRoutes,
) -> Int {
  self.session_lanes.length()
}

///|
fn LunaOnlinePublicationRoutes::try_bind(
  self : LunaOnlinePublicationRoutes,
  route : @core.SchedulerPublicationRoute,
  session_lane : Int,
) -> Bool {
  let route_lane = route.lane()
  if route_lane < 0 ||
    route_lane >= self.session_lanes.length() ||
    route.generation() == 0UL ||
    session_lane < 0 ||
    session_lane >= self.session_lanes.length() ||
    self.session_lanes[route_lane] != -1 ||
    self.generations[route_lane] != 0UL {
    return false
  }
  self.session_lanes[route_lane] = session_lane
  self.generations[route_lane] = route.generation()
  true
}

///|
fn LunaOnlinePublicationRoutes::bind(
  self : LunaOnlinePublicationRoutes,
  route : @core.SchedulerPublicationRoute,
  session_lane : Int,
) -> Unit {
  if !self.try_bind(route, session_lane) {
    abort("Luna online publication route bind invariant")
  }
}

///|
fn LunaOnlinePublicationRoutes::resolve(
  self : LunaOnlinePublicationRoutes,
  route : @core.SchedulerPublicationRoute,
) -> Int {
  self.lookups = route_counter_increment(self.lookups)
  if self.max_probes < 1 {
    self.max_probes = 1
  }
  let route_lane = route.lane()
  if route_lane < 0 ||
    route_lane >= self.session_lanes.length() ||
    route.generation() == 0UL ||
    self.generations[route_lane] != route.generation() {
    self.rejections = route_counter_increment(self.rejections)
    return -1
  }
  let session_lane = self.session_lanes[route_lane]
  if session_lane < 0 || session_lane >= self.session_lanes.length() {
    self.rejections = route_counter_increment(self.rejections)
    -1
  } else {
    session_lane
  }
}

///|
fn LunaOnlinePublicationRoutes::reject_authentication(
  self : LunaOnlinePublicationRoutes,
) -> Unit {
  self.rejections = route_counter_increment(self.rejections)
}

///|
fn LunaOnlinePublicationRoutes::try_clear(
  self : LunaOnlinePublicationRoutes,
  route : @core.SchedulerPublicationRoute,
  session_lane : Int,
) -> Bool {
  let route_lane = route.lane()
  if route_lane < 0 ||
    route_lane >= self.session_lanes.length() ||
    self.session_lanes[route_lane] != session_lane ||
    self.generations[route_lane] != route.generation() {
    return false
  }
  self.session_lanes[route_lane] = -1
  self.generations[route_lane] = 0UL
  true
}

///|
fn LunaOnlinePublicationRoutes::clear(
  self : LunaOnlinePublicationRoutes,
  route : @core.SchedulerPublicationRoute,
  session_lane : Int,
) -> Unit {
  if !self.try_clear(route, session_lane) {
    abort("Luna online publication route clear invariant")
  }
}
