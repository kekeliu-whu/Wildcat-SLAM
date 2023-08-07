#pragma once

#include <Eigen/Eigen>

template <int N>
Eigen::Matrix<double, N, 1> CubicBSplineApprox(
    const Eigen::Matrix<double, N, 1> &p_1,
    const Eigen::Matrix<double, N, 1> &p0,
    const Eigen::Matrix<double, N, 1> &p1,
    const Eigen::Matrix<double, N, 1> &p2,
    double                             s) {
  double s2 = s * s;
  double s3 = s * s * s;

  return (p_1 * std::pow(1 - s, 3) + p0 * (3 * s3 - 6 * s2 + 4) + p1 * (-3 * s3 + 3 * s2 + 3 * s + 1) + p2 * s3) / 6;
}

template <int N>
Eigen::Matrix<double, N, 1> CubicSplineInterpolate(
    double s_1, const Eigen::Matrix<double, N, 1> &p_1,
    double s0, const Eigen::Matrix<double, N, 1> &p0,
    double s1, const Eigen::Matrix<double, N, 1> &p1,
    double s2, const Eigen::Matrix<double, N, 1> &p2,
    double s) {
  Eigen::Matrix<double, N, 1> m0 = 0.5 * ((p0 - p_1) / (s0 - s_1) + (p1 - p0) / (s1 - s0));
  Eigen::Matrix<double, N, 1> m1 = 0.5 * ((p1 - p0) / (s1 - s0) + (p2 - p1) / (s2 - s1));

  double t  = (s - s0) / (s1 - s0);
  double t2 = t * t;
  double t3 = t * t * t;

  return (2 * t3 - 3 * t2 + 1) * p0 +
         (t3 - 2 * t2 + t) * (s1 - s0) * m0 +
         (-2 * t3 + 3 * t2) * p1 +
         (t3 - t2) * (s1 - s0) * m1;
}
