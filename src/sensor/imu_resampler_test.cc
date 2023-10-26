#include <gtest/gtest.h>

#include "imu_resampler.h"

using Vector1d = Eigen::Matrix<double, 1, 1>;

TEST(ImuResampler, Resampler) {
  auto    ir = ImuResampler(10);
  ImuData imu1{0, {1, 2, 3}, {435, 342, 434}};
  ImuData imu2{1, {11, 234, 453}, {234, 46, 32}};
  ir.AddImuData(imu1);
  ir.AddImuData(imu2);

  {
    auto resampled_imu = ir.AdvanceGetResampledImuData();
    EXPECT_TRUE(resampled_imu);
    EXPECT_EQ(resampled_imu->timestamp, 0);
  }
  {
    auto resampled_imu = ir.AdvanceGetResampledImuData();
    EXPECT_TRUE(resampled_imu);
    EXPECT_EQ(resampled_imu->timestamp, 0.1);
  }
  {
    auto resampled_imu = ir.AdvanceGetResampledImuData();
    EXPECT_TRUE(resampled_imu);
    EXPECT_EQ(resampled_imu->timestamp, 0.2);
    EXPECT_TRUE((0.8 * imu1.angular_velocity + 0.2 * imu2.angular_velocity).isApprox(resampled_imu->angular_velocity));
    EXPECT_TRUE((0.8 * imu1.linear_acceleration + 0.2 * imu2.linear_acceleration).isApprox(resampled_imu->linear_acceleration));
  }
}
