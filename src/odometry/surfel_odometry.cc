#include <glog/logging.h>
#include <pcl/io/ply_io.h>
#include <iomanip>

#include "odometry/surfel_odometry.h"
#include "surfel_extraction.h"

std::vector<std::pair<int, int>> BuildSurfelCorrespondences(const std::deque<Surfel> &surfel_list) {
  // todo kk
  return {};
}

void SurfelOdometry::AddLidarScan(const pcl::PointCloud<hilti_ros::Point>::Ptr &msg) {
  for (int i = 0; i < msg->size() - 1; ++i) {
    CHECK_GT(msg->points[i + 1].time, msg->points[i].time);
  }

  // 将点从雷达系转换至IMU系
  for (auto &e : *msg) {
    e.getVector3fMap() = this->pose_lidar2imu_.cast<float>() * e.getVector3fMap();
  }

  // undistort points of a scan
  std::vector<Surfel> surfels;
  BuildSurfels(msg, surfels);

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

void SurfelOdometry::SetExtrinsicLidar2Imu(const Rigid3d &pose) {
  this->pose_lidar2imu_ = pose;
}
