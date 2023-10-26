#pragma once

#include <deque>

#include "common/common.h"

/**
 * @brief Sample IMU data to a fixed frequency sequence
 * 
 */
class ImuResampler {
 public:
  ImuResampler(int freq) : freq_(freq), is_first_sample_(true) {
  }

  void AddImuData(const ImuData& imu_data) {
    imu_queue_.push_back(imu_data);
    if (imu_queue_.size() > 2) {
      imu_queue_.pop_front();
    }
  }

  std::shared_ptr<ImuData> AdvanceGetResampledImuData() {
    if (imu_queue_.size() == 2) {
      if (is_first_sample_) {  // check if the first sample
        prev_sample_time_ = imu_queue_[0].timestamp;
        is_first_sample_  = false;
        return std::make_shared<ImuData>(imu_queue_[0]);
      }

      double target_sample_time = prev_sample_time_ + (1.0 / freq_);
      if (imu_queue_[0].timestamp <= target_sample_time && target_sample_time <= imu_queue_[1].timestamp) {
        double factor = (target_sample_time - imu_queue_[0].timestamp) / (imu_queue_[1].timestamp - imu_queue_[0].timestamp);

        ImuData sampled_imu_data;
        sampled_imu_data.timestamp           = target_sample_time;
        sampled_imu_data.linear_acceleration = (1 - factor) * imu_queue_[0].linear_acceleration + factor * imu_queue_[1].linear_acceleration;
        sampled_imu_data.angular_velocity    = (1 - factor) * imu_queue_[0].angular_velocity + factor * imu_queue_[1].angular_velocity;

        prev_sample_time_ = target_sample_time;
        return std::make_shared<ImuData>(sampled_imu_data);
      }
    }

    return nullptr;
  }

 private:
  std::deque<ImuData> imu_queue_;
  int                 freq_;

  double prev_sample_time_;
  bool   is_first_sample_;
};
