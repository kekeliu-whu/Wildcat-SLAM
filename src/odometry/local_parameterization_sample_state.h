#pragma once

#include <ceres/ceres.h>

#include "sophus/so3.hpp"

struct LocalParameterizationSampleState {
  template <typename T>
  bool operator()(const T* x_ptr, const T* delta_ptr, T* x_plus_delta_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> x_so3{x_ptr};
    Eigen::Map<const Eigen::Matrix<T, 9, 1>> x_other{x_ptr + 3};

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> delta_so3{delta_ptr};
    Eigen::Map<const Eigen::Matrix<T, 9, 1>> delta_other{delta_ptr + 3};

    Eigen::Map<Eigen::Matrix<T, 3, 1>> x_plus_delta_so3{x_plus_delta_ptr};
    Eigen::Map<Eigen::Matrix<T, 9, 1>> x_plus_delta_other{x_plus_delta_ptr + 3};

    x_plus_delta_so3   = (Sophus::SO3<T>::exp(x_so3) * Sophus::SO3<T>::exp(delta_so3)).log();
    x_plus_delta_other = x_other + delta_other;

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return new ceres::AutoDiffLocalParameterization<LocalParameterizationSampleState, 12, 12>();
  }
};

struct LocalParameterizationSampleStateFixPos {
  template <typename T>
  bool operator()(const T* x_ptr, const T* delta_ptr, T* x_plus_delta_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> x_so3{x_ptr};
    Eigen::Map<const Eigen::Matrix<T, 9, 1>> x_other{x_ptr + 3};

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> delta_so3{delta_ptr};
    Eigen::Map<const Eigen::Matrix<T, 9, 1>> delta_other{delta_ptr + 3};

    Eigen::Map<Eigen::Matrix<T, 3, 1>> x_plus_delta_so3{x_plus_delta_ptr};
    Eigen::Map<Eigen::Matrix<T, 9, 1>> x_plus_delta_other{x_plus_delta_ptr + 3};

    x_plus_delta_so3                              = (Sophus::SO3<T>::exp(x_so3) * Sophus::SO3<T>::exp(delta_so3)).log();
    x_plus_delta_other.template block<3, 1>(0, 0) = x_other.template block<3, 1>(0, 0);
    x_plus_delta_other.template block<6, 1>(3, 0) = x_other.template block<6, 1>(3, 0) + delta_other.template block<6, 1>(3, 0);

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return new ceres::AutoDiffLocalParameterization<LocalParameterizationSampleState, 12, 12>();
  }
};
