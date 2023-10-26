#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sophus/so3.hpp>

#include "common/histogram.h"
#include "spline_interpolation.h"

using Vector1d = Eigen::Matrix<double, 1, 1>;

TEST(BSplineApproxTest, ArithmeticProgression) {
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{1}, Vector1d{2}, Vector1d{3}, Vector1d{4}, 0)[0], 2);
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{1}, Vector1d{2}, Vector1d{3}, Vector1d{4}, 1)[0], 3);
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{1}, Vector1d{2}, Vector1d{3}, Vector1d{4}, 0.4)[0], 2.4);
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{1}, Vector1d{2}, Vector1d{3}, Vector1d{4}, 0.5)[0], 2.5);
}

TEST(BSplineApproxTest, Const) {
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{2}, Vector1d{2}, Vector1d{2}, Vector1d{2}, 0)[0], 2);
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{2}, Vector1d{2}, Vector1d{2}, Vector1d{2}, 1)[0], 2);
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{2}, Vector1d{2}, Vector1d{2}, Vector1d{2}, 0.5)[0], 2);
  EXPECT_DOUBLE_EQ(CubicBSplineApprox(Vector1d{2}, Vector1d{2}, Vector1d{2}, Vector1d{2}, 0.4)[0], 2);
}

TEST(SplineInterpolateTest, ArithmeticProgression) {
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{1}, 0, Vector1d{2}, 1, Vector1d{3}, 2, Vector1d{4}, 0)[0], 2);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{1}, 0, Vector1d{2}, 1, Vector1d{3}, 2, Vector1d{4}, 1)[0], 3);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{1}, 0, Vector1d{2}, 1, Vector1d{3}, 2, Vector1d{4}, 0.4)[0], 2.4);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{1}, 0, Vector1d{2}, 1, Vector1d{3}, 2, Vector1d{4}, 0.5)[0], 2.5);
}

TEST(SplineInterpolateTest, Const) {
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{2}, 0, Vector1d{2}, 1, Vector1d{2}, 2, Vector1d{2}, 0)[0], 2);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{2}, 0, Vector1d{2}, 1, Vector1d{2}, 2, Vector1d{2}, 1)[0], 2);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{2}, 0, Vector1d{2}, 1, Vector1d{2}, 2, Vector1d{2}, 0.5)[0], 2);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{2}, 0, Vector1d{2}, 1, Vector1d{2}, 2, Vector1d{2}, 0.4)[0], 2);
}

TEST(SplineInterpolateTest, BeginEnd) {
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{2}, 0, Vector1d{3}, 1, Vector1d{1}, 2, Vector1d{2}, 0)[0], 3);
  EXPECT_DOUBLE_EQ(CubicSplineInterpolate(-1, Vector1d{2}, 0, Vector1d{3}, 1, Vector1d{1}, 2, Vector1d{2}, 1)[0], 1);
}

TEST(BSplineTest, 1) {
  // for (double d = 0; d <= 1.01; d += 0.02) {
  //   auto ret = CubicBSpline(Vector1d{4}, Vector1d{3}, Vector1d{2}, Vector1d{1}, d);
  //   std::cout << d << " " << ret << std::endl;
  // }
}

TEST(SO3InterpolationTest, RotInterpolation1) {
  using V3D = Vector3d;

  // If both rotations are less than 10 degrees, the error in the angle
  // obtained by linear interpolation relative to slerp is less than 0.04 degrees.

  const double kRad2Deg = 180 / M_PI;
  double       s        = 0.5;

  Histogram hist;

  for (int i = 0; i < 100'000; ++i) {
    V3D ra = V3D::Random() * 10 / kRad2Deg;
    V3D rb = V3D::Random() * 10 / kRad2Deg;

    auto Ra = Sophus::SO3d::exp(ra);
    auto Rb = Sophus::SO3d::exp(rb);

    auto R_real   = Sophus::SO3d(Ra.unit_quaternion().slerp(s, Rb.unit_quaternion()));
    auto R_approx = Sophus::SO3d::exp((1 - s) * ra + s * rb);

    auto r_err = R_real.unit_quaternion().angularDistance(R_approx.unit_quaternion()) * kRad2Deg;

    hist.Add(r_err);
  }

  LOG(INFO) << hist.ToString(10);
}

TEST(CubicBSplineInterpolator, InterpolationTest) {
  // Sample data point
  std::vector<double>   timestamps{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
  std::vector<Vector3d> points;

  Eigen::MatrixXd p(8, 3);
  p << 1, 1, 1, 2, 3, 2, 4, 5, 5, 6, 6, 3, 5, 4, 1, 6, 7, 1, 9, 9, 8, 12, 15, 11;
  for (int i = 0; i < p.rows(); ++i) {
    points.push_back(Vector3d(p(i, 0), p(i, 1), p(i, 2)));
  }

  CubicBSplineInterpolator interpolator(timestamps, points);
  for (int i = 0; i < timestamps.size(); ++i) {
    auto p = interpolator.Interp(timestamps[i]);
    EXPECT_TRUE(p);
    EXPECT_TRUE(p->isApprox(points[i], 1e-6));
  }
}
