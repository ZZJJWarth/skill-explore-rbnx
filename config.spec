# Runtime config accepted by the Explore skill.
#
# The package accepts the deployment's inherited reference. Exploration limits
# belong to each robonix/skill/explore/explore request:
#
# - area_hint: string describing the requested exploration area.
# - timeout_s: unsigned integer execution deadline in seconds.
# - max_speed_m_s: floating-point linear speed limit in metres per second.
#
# This file is documentation only and is not loaded by the provider.

config:
  reference:
    kind: linux_system  # linux_system | ros_time
    provider_id: chronos  # used for ros_time
