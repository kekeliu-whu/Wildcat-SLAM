#pragma once

#include <ceres/ceres.h>
#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud2.h>
#include <deque>

#include "common/common.h"
#include "io/odometry_io.h"
#include "odometry/lio_config.h"
#include "odometry/surfel.h"

struct MeasureGroup {
  int                          sweep_id;
  double                       sweep_beg_time;
  double                       sweep_end_time;
  std::deque<ImuData>          imu_msgs;
  std::deque<hilti_ros::Point> lidar_points;
};

class LidarOdometry {
 public:
  LidarOdometry();

  ~LidarOdometry();

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

 private:
  /**
   * @brief Remove heading imus and points to make sure they are in sync
   *
   * @return true
   * @return false
   */
  bool SyncHeadingMsgs();

  bool SyncPackages(MeasureGroup &mg);

  void BuildSldWinLidarResiduals(const std::vector<SurfelCorrespondence> &surfel_corrs, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids);

  void BuildFixWinLidarResiduals(const std::vector<SurfelCorrespondence> &surfel_corrs, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids);

  void BuildImuResiduals(const std::deque<ImuState> &imu_states, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids);

 private:
  LioConfig config_;

  std::deque<Surfel::Ptr>      surfels_sld_win_;
  std::deque<Surfel::Ptr>      surfels_fix_win_;
  std::deque<SampleState::Ptr> sample_states_sld_win_;
  std::deque<ImuState>         imu_states_sld_win_;

  std::deque<ImuData>          imu_buff_;
  std::deque<hilti_ros::Point> points_buff_;
  std::deque<hilti_ros::Point> points_buff_sld_win_;

  ros::NodeHandle nh_;
  ros::Publisher  pub_plane_map_;
  ros::Publisher  pub_scan_in_imu_frame_;
  ros::Publisher  pub_imu_path_;

  std::unique_ptr<ceres::LossFunction> surfel_match_cauchy_loss_;

  int sweep_id_ = 0;

  OdometryIO io_;
};
