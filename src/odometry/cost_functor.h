#pragma once

#include <ceres/ceres.h>

#include "odometry/surfel.h"

template <typename T>
using Vector12 = Eigen::Matrix<T, 12, 1>;

/**
 * @brief s1 s2 is two corresponding surfels
 *
 * Timestamp order: s1 < sp2l <= s2 < sp2r
 *
 */
struct SurfelMatchUnaryFactor {
  SurfelMatchUnaryFactor(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r);

  template <typename T>
  bool operator()(const T* sp2l_ptr, const T* sp2r_ptr, T* residuals) const;

  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r);

 private:
  std::shared_ptr<Surfel>      s1_;
  std::shared_ptr<SampleState> sp2l_;
  std::shared_ptr<SampleState> sp2r_;
  std::shared_ptr<Surfel>      s2_;

  Vector3d norm_;
  double   weight_;
};

/**
 * @brief s1 s2 is two corresponding surfels
 *
 */
struct SurfelMatchBinaryFactor {
  SurfelMatchBinaryFactor(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r);

  template <typename T>
  bool operator()(const T* sp1l_ptr, const T* sp1r_ptr, const T* sp2l_ptr, const T* sp2r_ptr, T* residuals) const;

  template <typename T>
  bool operator()(const T* sp1l_ptr, const T* sp1r_ptr, const T* sp2r_ptr, T* residuals) const;

  template <typename T>
  bool operator()(const T* sp1l_ptr, const T* sp1r_ptr, T* residuals) const;

  // sp1l <= s1 < sp1r < sp2l <= s2 < sp2r
  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r);

  // sp1l <= s1 < sp1r = sp2l <= s2 < sp2r
  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2r);

  // sp1l = sp2l <= s1 < s2 < sp1r = sp2r
  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2);

 private:
  template <typename T>
  void Helper(const T* sp1l_ptr, const T* sp1r_ptr, const T* sp2l_ptr, const T* sp2r_ptr, T* residuals) const;

 private:
  std::shared_ptr<Surfel>      s1_;
  std::shared_ptr<SampleState> sp1l_;
  std::shared_ptr<SampleState> sp1r_;
  std::shared_ptr<Surfel>      s2_;
  std::shared_ptr<SampleState> sp2l_;
  std::shared_ptr<SampleState> sp2r_;

  Vector3d norm_;
  double   weight_;
};

/**
 * @brief
 *
 * Timestamp order: sp1 <= i1 < sp2 and i1 < i2 < i3 and sp1 < sp2 < sp3
 */
struct ImuFactorWith3SampleStates {
  ImuFactorWith3SampleStates(const ImuState& i1, const ImuState& i2, const ImuState& i3,
                             double sp1_timestamp, double sp2_timestamp, double sp3_timestamp,
                             double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
                             double dt, const Vector3d& gravity);

  template <typename T>
  bool operator()(const T* sp1_ptr, const T* sp2_ptr, const T* sp3_ptr, T* residuals) const;

  static ceres::CostFunction* Create(const ImuState& i1, const ImuState& i2, const ImuState& i3,
                                     double sp1_timestamp, double sp2_timestamp, double sp3_timestamp,
                                     double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
                                     double dt, const Vector3d& gravity);

 private:
  template <typename T>
  void ComputeStateCorr(
      Eigen::Map<const Vector12<T>>& sp1,
      Eigen::Map<const Vector12<T>>& sp2,
      Eigen::Map<const Vector12<T>>& sp3,
      double                         sp1_timestamp,
      double                         sp2_timestamp,
      double                         sp3_timestamp,
      double                         timestamp,
      Eigen::Matrix<T, 3, 1>&        r_cor,
      Eigen::Matrix<T, 3, 1>&        t_cor,
      Eigen::Matrix<T, 3, 1>&        bg,
      Eigen::Matrix<T, 3, 1>&        ba) const;

 private:
  ImuState i1_, i2_, i3_;
  double   sp1_timestamp_, sp2_timestamp_, sp3_timestamp_;

  double weight_gyr_;
  double weight_acc_;
  double weight_bg_;
  double weight_ba_;

  double dt_;

  Vector3d gravity_;
};

/**
 * @brief
 *
 * Timestamp order: sp1 <= i1 < i2 < i3 <= sp2
 */
struct ImuFactorWith2SampleStates {
  ImuFactorWith2SampleStates(const ImuState& i1, const ImuState& i2, const ImuState& i3,
                             double sp1_timestamp, double sp2_timestamp, double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
                             double dt, const Vector3d& gravity);

  template <typename T>
  bool operator()(const T* sp1_ptr, const T* sp2_ptr, T* residuals) const;

  static ceres::CostFunction* Create(const ImuState& i1, const ImuState& i2, const ImuState& i3,
                                     double sp1_timestamp, double sp2_timestamp,
                                     double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
                                     double dt, const Vector3d& gravity);

 private:
  template <typename T>
  void ComputeStateCorr(
      const Eigen::Map<const Vector12<T>>& sp1,
      const Eigen::Map<const Vector12<T>>& sp2,
      double                               sp1_timestamp,
      double                               sp2_timestamp,
      double                               timestamp,
      Eigen::Matrix<T, 3, 1>&              r_cor,
      Eigen::Matrix<T, 3, 1>&              t_cor,
      Eigen::Matrix<T, 3, 1>&              bg,
      Eigen::Matrix<T, 3, 1>&              ba) const;

 private:
  ImuState i1_, i2_, i3_;
  double   sp1_timestamp_, sp2_timestamp_, sp3_timestamp_;

  double weight_gyr_;
  double weight_acc_;
  double weight_bg_;
  double weight_ba_;

  double dt_;

  Vector3d gravity_;
};
