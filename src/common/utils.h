#pragma once

#include <Eigen/Dense>

#include "common/common.h"
#include "sophus/so3.hpp"

template <class scalar>
inline scalar tolerance();
template <>
inline float tolerance<float>() { return 1e-5f; }
template <>
inline double tolerance<double>() { return 1e-10; }

template <typename Base>
Eigen::Matrix<typename Base::Scalar, 3, 3> Hat(const Base& v) {
  Eigen::Matrix<typename Base::Scalar, 3, 3> res;
  res << 0, -v[2], v[1],
      v[2], 0, -v[0],
      -v[1], v[0], 0;
  return res;
}

template <typename T>
inline Eigen::Quaternion<T> Exp(const Eigen::Matrix<T, 3, 1>& v) {
  return Sophus::SO3<T>::exp(v).unit_quaternion();
}

inline Quaterniond Exp(const Vector3d& v) {
  return Sophus::SO3d::exp(v).unit_quaternion();
}

template <typename T>
inline Eigen::Matrix<T, 3, 1> Log(const Eigen::Quaternion<T>& q) {
  return Sophus::SO3<T>(q).log();
}

template <typename Base>
Eigen::Matrix<typename Base::Scalar, 3, 3> Jl_inv(const Base& v) {
  Eigen::Matrix<typename Base::Scalar, 3, 3> res;
  if (v.norm() > ::tolerance<typename Base::Scalar>()) {
    res = Eigen::Matrix<typename Base::Scalar, 3, 3>::Identity() - 0.5 * Hat<Base>(v) + (1 - v.norm() * std::cos(v.norm() / 2) / 2 / std::sin(v.norm() / 2)) * Hat(v) * Hat(v) / v.squaredNorm();
  } else {
    res = Eigen::Matrix<typename Base::Scalar, 3, 3>::Identity();
  }

  return res;
}

template <typename Base>
Eigen::Matrix<typename Base::Scalar, 3, 3> Jl(const Base& v) {
  Eigen::Matrix<typename Base::Scalar, 3, 3> res;
  if (v.norm() > ::tolerance<typename Base::Scalar>()) {
    typename Base::Scalar                      theta = v.norm();
    Eigen::Matrix<typename Base::Scalar, 3, 1> a     = v / theta;
    res                                              = sin(theta) / theta * Eigen::Matrix<typename Base::Scalar, 3, 3>::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * Hat(a);
  } else {
    res = Eigen::Matrix<typename Base::Scalar, 3, 3>::Identity();
  }

  return res;
}

template <typename Base>
Eigen::Matrix<typename Base::Scalar, 3, 3> Jr(const Base& v) {
  // return Jl(v).transpose();
  return Jl(-v);
}

template <typename Base>
Eigen::Matrix<typename Base::Scalar, 3, 3> Jr_inv(const Base& v) {
  return Jl_inv(-v);
}
