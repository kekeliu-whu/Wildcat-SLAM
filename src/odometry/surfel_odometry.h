#pragma once

#include <deque>
#include "common/rigid_transform.h"

struct Surfel {
  double           timestamp;
  double           resolution;
  Eigen::Vector3d  center;
  Eigen::Vector3d  norm;
  Eigen::Matrix3cd covariance;
};

struct TimestampedPose {
  double  timestamp;
  Rigid3d pose;
};

class SurfelOdometry {
 public:
  SurfelOdometry(
      double sample_dt,
      int    sliding_window_size,
      int    marginalization_stride)
      : sample_dt_(sample_dt),
        sliding_window_size_(sliding_window_size),
        marginalization_stride_(marginalization_stride) {
  }

  /**
   * @brief Add raw imu measurements to queue
   *
   */
  void AddImuData(const ImuData &msg);

  /**
   * @brief Add raw lidar points with timestamp
   *
   */
  void AddLidarPoints();

 private:
  bool CubicBSplineInterpolate(double timestamp, Rigid3d &pose);

  bool UpdateSamplePoses();

  bool UpdateImuPoses();

 private:
  double sample_dt_;
  int    sliding_window_size_;
  int    marginalization_stride_;

  std::deque<Surfel>          surfel_list_;
  std::deque<TimestampedPose> sampled_poses_;
  std::deque<TimestampedPose> sampled_poses_corr_;
};
