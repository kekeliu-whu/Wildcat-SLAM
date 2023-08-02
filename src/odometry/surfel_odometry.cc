#include <glog/logging.h>

#include "odometry/surfel_odometry.h"

std::vector<std::pair<int, int>> BuildSurfelCorrespondences(const std::deque<Surfel> &surfel_list) {
  // todo kk
  return {};
}

void SurfelOdometry::AddLidarPoints() {
  // Collect imu and lidar measurements
  // predict pose by imu integration
  // project lidar points into world frame

  // if marginalization_stride_ reached
  // 1. extract multiple resolution surfels from newly received points
  // 2. build surfel correspondences
  // 3. begin local optimisation to estimate pose correction
  // 4. update sampled poses
  // 5. update IMU poses
  // 6. update surfel postion
}

bool CubicBSplineInterpolate(double timestamp, Rigid3d &pose) {
  return 0;
}

void SurfelOdometry::AddImuData(const ImuData &msg) {
}

bool SurfelOdometry::UpdateSamplePoses() {
  CHECK_EQ(this->sampled_poses_.size(), sampled_poses_corr_.size());

  for (int i = 0; i < this->sampled_poses_.size(); ++i) {
    // todo kk
    sampled_poses_[i].pose = sampled_poses_corr_[i].pose * sampled_poses_[i].pose;
  }

  return true;
}

bool SurfelOdometry::UpdateImuPoses() {
  return true;
}
