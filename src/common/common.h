#pragma once

#include <pcl/point_types.h>

#include "common/rigid_transform.h"

using Vector3d    = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;
using Matrix3d    = Eigen::Matrix3d;

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
  double   timestamp;
  Vector3d linear_acceleration;
  Vector3d angular_velocity;
};

using PointType = hilti_ros::Point;
