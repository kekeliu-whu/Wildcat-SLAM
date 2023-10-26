#pragma once

#include <Eigen/Eigen>
#include <cmath>

#include "common/rigid_transform.h"

struct LioConfig {
 private:
  double gyroscope_noise_density     = 0.00015198973532354657;
  double accelerometer_noise_density = 0.006308226052016165;
  double gyroscope_random_walk       = 0.00011673723527962174;
  double accelerometer_random_walk   = 2.664506559330434e-06;
  double imu_factor_weight           = 0.01;  // a weight factor between imu factor and lidar factor

 public:
  ///////////////////// Preprocess parameters //////////////////////
  double                       max_range = 120;
  double                       min_range = 0.3;
  Eigen::AlignedBox<double, 3> blind_bounding_box{
      Eigen::Vector3d{-0.8, -0.5, -0.4},
      Eigen::Vector3d{0.3, 0.5, 0.4}};  // in imu_link
  Rigid3d ext_lidar2imu{
      Eigen::Vector3d(-0.001, -0.00855, 0.055),
      Eigen::Quaterniond(
          (Eigen::Matrix3d() << -5.32125e-08, -1, 0,
           -1, -5.32125e-08, -0,
           0, 0, -1)
              .finished())};

  ///////////////////// Sliding window preprocess parameters //////////////////////
  double imu_rate                = 200;   // imu rate in Hz
  double sample_dt               = 0.08;  // sample time in seconds
  double fixed_window_duration   = 3.0;   // fixed window duration in seconds // todo kk change duration to size
  double sliding_window_duration = 6.0;   // sliding window duration in seconds
  double sweep_duration          = 0.5;   // sweep duration in seconds

  ///////////////////// Sliding windows optimization parameters //////////////////////
  double gravity_norm                            = 9.81;
  int    outer_iter_num_max                      = 1;
  int    inner_iter_num_max                      = 100;
  double gyroscope_noise_density_cost_weight     = 1 / (gyroscope_noise_density * sqrt(imu_rate)) * imu_factor_weight;
  double accelerometer_noise_density_cost_weight = 1 / (accelerometer_noise_density * sqrt(imu_rate)) * imu_factor_weight;
  double gyroscope_random_walk_cost_weight       = 1 / (gyroscope_random_walk / sqrt(imu_rate)) * imu_factor_weight;
  double accelerometer_random_walk_cost_weight   = 1 / (accelerometer_random_walk / sqrt(imu_rate)) * imu_factor_weight;
};
