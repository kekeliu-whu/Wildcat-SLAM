#pragma once

#include <glog/logging.h>
#include <Eigen/Eigen>
#include <memory>

#include "common/common.h"

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

class CubicBSplineInterpolator {
 public:
  CubicBSplineInterpolator(const std::vector<double>   &timestamps,
                           const std::vector<Vector3d> &points)
      : timestamps_(timestamps), points_(points) {
    CHECK_EQ(timestamps.size(), points.size());
    Init();
  }

  std::shared_ptr<Eigen::Matrix<double, 3, 1>> Interp(double timestamp) {
    if (timestamp < timestamps_.front() || timestamp > timestamps_.back()) {
      return nullptr;
    }

    double index_f   = (timestamp - timestamps_.front()) / (timestamps_.back() - timestamps_.front()) * (timestamps_.size() - 1) + 1.0;
    int    index_int = std::floor(index_f);
    double t         = index_f - index_int;

    Eigen::Array4i indexV = Eigen::ArrayXi::LinSpaced(4, index_int - 2, index_int + 1);
    indexV                = indexV.max(0).min(Np - 1);
    Eigen::Vector4d tv    = Eigen::Vector4d(t * t * t, t * t, t, 1.0);

    Eigen::Matrix<double, 4, 3> Q_4;
    for (int j = 0; j < 4; ++j) {
      Q_4.row(j) = Q.row(indexV[j]);
    }
    Vector3d pbs = tv.transpose() * M * Q_4 / 6.0;

    return std::make_shared<Eigen::Matrix<double, 3, 1>>(pbs);
  }

 private:
  void Init() {
    Eigen::MatrixXd p(points_.size(), 3);
    for (int i = 0; i < points_.size(); ++i) {
      p.row(i) = points_[i].transpose();
    }

    Np = p.rows();

    // Basis matrix
    M << -1, 3, -3, 1, 3, -6, 3, 0, -3, 0, 3, 0, 1, 4, 1, 0;

    // Calculate the best control point locations
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(Np, Np);

    for (int i = 0; i < Np; ++i) {
      int index_i = i;

      Eigen::ArrayXi indexV = Eigen::ArrayXi::LinSpaced(4, index_i - 1, index_i + 2);
      indexV                = indexV.max(0).min(Np - 1);
      Eigen::Vector4d tv    = Eigen::Vector4d(0, 0, 0, 1.0);
      Eigen::Vector4d temp  = tv.transpose() * M / 6.0;

      for (int j = 0; j < 4; ++j) {
        N(i, indexV[j]) += temp[j];
      }
    }

    // Solve linear system to obtain control points
    Eigen::MatrixXd invNN_Nt = (N.transpose() * N).inverse() * N.transpose();
    Q                        = invNN_Nt * p;
  }

 private:
  std::vector<double>   timestamps_;
  std::vector<Vector3d> points_;

  int             Np;
  Eigen::MatrixXd Q;
  Eigen::Matrix4d M;
};
