#pragma once

#include <ceres/ceres.h>
#include <ros/node_handle.h>
#include <sensor_msgs/PointCloud2.h>
#include <deque>

#include "odometry/lio_config.h"
#include "surfel_extraction.h"

class LidarOdometry {
 public:
  LidarOdometry();

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
   * @brief Predict imu states and sample states
   *
   * Timestamp Order:
   *   IMU:          i_0 < i_1 < ... < i_{n-2} < end_time <= i_{n-1}
   *   Sample State: s_0 < s_1 < ... < s_{n-1} < end_time
   *
   * @param end_time
   */
  void PredictImuStatesAndSampleStates(double end_time);

  /**
   * @brief Remove heading imus and points to make sure they are in sync
   *
   * @return true
   * @return false
   */
  bool SyncHeadingMsgs();

  void BuildLidarResiduals(const std::vector<SurfelCorrespondence> &surfel_corrs, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids);

  void BuildImuResiduals(const std::deque<ImuState> &imu_states, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids);

 private:
  LioConfig config_;

  std::deque<Surfel::Ptr>      surfels_sld_win_;
  std::deque<SampleState::Ptr> sample_states_sld_win_;
  std::deque<ImuState>         imu_states_sld_win_;

  std::deque<ImuData>          imu_buff_;
  std::deque<hilti_ros::Point> points_buff_;

  ros::NodeHandle nh_;
  ros::Publisher  pub_plane_map_;
  ros::Publisher  pub_scan_in_imu_frame_;

  int sweep_id_ = 0;
};
