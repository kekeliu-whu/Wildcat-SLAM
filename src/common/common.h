#pragma once

#include <pcl/point_types.h>

#include "common/rigid_transform.h"
#include "common/time.h"

using Vector3d    = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;

struct State {
  double          timestamp;
  Rigid3d         pose;
  Eigen::Vector3d bias_a;
  Eigen::Vector3d bias_g;
  Eigen::Vector3d gravity;
};

// clang-format off
namespace hilti_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      double time;
      std::uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace hilti_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(hilti_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (double, time, timestamp)
    (std::uint16_t, ring, ring)
)
// clang-format on

struct ImuData {
  Time            time;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};
