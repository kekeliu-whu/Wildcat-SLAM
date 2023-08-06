#pragma once

#include <deque>

#include "surfel_extraction.h"

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
  void AddLidarScan(const pcl::PointCloud<hilti_ros::Point>::Ptr &msg);

  void SetExtrinsicLidar2Imu(const Rigid3d &pose);

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

  Rigid3d pose_lidar2imu_;
};
