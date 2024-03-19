#pragma once

#include <ceres/ceres.h>
#include <iomanip>

#include "common/utils.h"
#include "odometry/surfel.h"

template <typename T>
using Vector12 = Eigen::Matrix<T, 12, 1>;

/**
 * @brief s1 s2 is two corresponding surfels
 *
 * 1. Timestamp order: s1 < sp2l <= s2 < sp2r
 * 2. s1 s2 are not in adjacent sampled intervals
 *
 */
struct SurfelMatchUnaryFactor {
  SurfelMatchUnaryFactor(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r) : s1_(s1), sp2l_(sp2l), sp2r_(sp2r), s2_(s2) {
    Matrix3d                                cov = s1_->GetCovarianceInWorld() + s2_->GetCovarianceInWorld();
    Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
    weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
    norm_   = es.eigenvectors().col(0);
  }

  template <typename T>
  bool operator()(const T* sp2l_ptr, const T* sp2r_ptr, T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_sp2l{sp2l_ptr};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_sp2l{sp2l_ptr + 3};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_sp2r{sp2r_ptr};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_sp2r{sp2r_ptr + 3};
    double                                   factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
    Eigen::Matrix<T, 3, 1>                   r_s2    = (1 - factor2) * r_sp2l + factor2 * r_sp2r;
    Eigen::Matrix<T, 3, 1>                   t_s2    = (1 - factor2) * t_sp2l + factor2 * t_sp2r;
    CHECK_GE(factor2, 0);
    CHECK_LE(factor2, 1);

    residuals[0] = weight_ * norm_.cast<T>().dot(s1_->GetCenterInWorld().cast<T>() - Exp(r_s2) * s2_->rot.cast<T>() * s2_->CenterInBody().cast<T>() - t_s2 - s2_->pos.cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r) {
    return new ceres::AutoDiffCostFunction<SurfelMatchUnaryFactor, 1, 12, 12>(new SurfelMatchUnaryFactor(s1, s2, sp2l, sp2r));
  }

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
 * 1. Timestamp order:
 *   a. Mode 0: sp1l <= s1 < sp1r < sp2l <= s2 < sp2r
 *   b. Mode 1: sp1l <= s1 < sp1r = sp2l <= s2 < sp2r
 *   c. Mode 2: sp1l = sp2l <= s1 < s2 < sp1r = sp2r
 * 2. s1 s2 are not in adjacent sampled intervals
 *
 */
struct SurfelMatchBinaryFactor {
  SurfelMatchBinaryFactor(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r) : s1_(s1), sp1l_(sp1l), sp1r_(sp1r), s2_(s2), sp2l_(sp2l), sp2r_(sp2r) {
    Matrix3d                                cov = s1_->GetCovarianceInWorld() + s2_->GetCovarianceInWorld();
    Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
    weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
    norm_   = es.eigenvectors().col(0);
  }

  template <typename T>
  bool operator()(const T* sp1l_ptr, const T* sp1r_ptr, const T* sp2l_ptr, const T* sp2r_ptr, T* residuals) const {
    Helper(sp1l_ptr, sp1r_ptr, sp2l_ptr, sp2r_ptr, residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* sp1l_ptr, const T* sp1r_ptr, const T* sp2r_ptr, T* residuals) const {
    Helper(sp1l_ptr, sp1r_ptr, sp1r_ptr, sp2r_ptr, residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* sp1l_ptr, const T* sp1r_ptr, T* residuals) const {
    Helper(sp1l_ptr, sp1r_ptr, sp1l_ptr, sp1r_ptr, residuals);
    return true;
  }

  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r) {
    CHECK_LT(s1->timestamp, s2->timestamp);
    CHECK_LT(sp1r->timestamp, sp2l->timestamp);
    return new ceres::AutoDiffCostFunction<SurfelMatchBinaryFactor, 1, 12, 12, 12, 12>(
        new SurfelMatchBinaryFactor(s1, sp1l, sp1r, s2, sp2l, sp2r));
  }

  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2r) {
    CHECK_LT(s1->timestamp, s2->timestamp);
    return new ceres::AutoDiffCostFunction<SurfelMatchBinaryFactor, 1, 12, 12, 12>(
        new SurfelMatchBinaryFactor(s1, sp1l, sp1r, s2, sp1r, sp2r));
  }

  static ceres::CostFunction* Create(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2) {
    CHECK_LT(s1->timestamp, s2->timestamp);
    return new ceres::AutoDiffCostFunction<SurfelMatchBinaryFactor, 1, 12, 12>(
        new SurfelMatchBinaryFactor(s1, sp1l, sp1r, s2, sp1l, sp1r));
  }

 private:
  template <typename T>
  void Helper(const T* sp1l_ptr, const T* sp1r_ptr, const T* sp2l_ptr, const T* sp2r_ptr, T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_sp1l{sp1l_ptr};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_sp1l{sp1l_ptr + 3};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_sp1r{sp1r_ptr};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_sp1r{sp1r_ptr + 3};
    double                                   factor1 = (s1_->timestamp - sp1l_->timestamp) / (sp1r_->timestamp - sp1l_->timestamp);
    Eigen::Matrix<T, 3, 1>                   r_s1    = (1 - factor1) * r_sp1l + factor1 * r_sp1r;
    Eigen::Matrix<T, 3, 1>                   t_s1    = (1 - factor1) * t_sp1l + factor1 * t_sp1r;
    CHECK_GE(factor1, 0);
    CHECK_LE(factor1, 1);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_sp2l{sp2l_ptr};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_sp2l{sp2l_ptr + 3};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r_sp2r{sp2r_ptr};
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_sp2r{sp2r_ptr + 3};
    double                                   factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
    Eigen::Matrix<T, 3, 1>                   r_s2    = (1 - factor2) * r_sp2l + factor2 * r_sp2r;
    Eigen::Matrix<T, 3, 1>                   t_s2    = (1 - factor2) * t_sp2l + factor2 * t_sp2r;
    CHECK_GE(factor2, 0);
    CHECK_LE(factor2, 1);

    residuals[0] = weight_ * norm_.cast<T>().dot(Exp(r_s1) * s1_->rot.cast<T>() * s1_->CenterInBody().cast<T>() + t_s1 + s1_->pos.cast<T>() - Exp(r_s2) * s2_->rot.cast<T>() * s2_->CenterInBody().cast<T>() - t_s2 - s2_->pos.cast<T>());
  }

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

template <int Mode>
struct ImuFactorModeTraits {
};

template <>
struct ImuFactorModeTraits<0> {
  using type = ceres::SizedCostFunction<12, 12, 12, 12>;
};

template <>
struct ImuFactorModeTraits<1> {
  using type = ceres::SizedCostFunction<12, 12, 12>;
};

/**
 * @brief
 *
 * Timestamp order:
 *   a. Mode 0: sp1 <= i1 < sp2 and i1 < i2 < i3 and sp1 < sp2 < sp3
 *   b. Mode 1: sp1 <= i1 < i2 < i3 <= sp2
 */
template <int Mode, typename TMode = typename ImuFactorModeTraits<Mode>::type>
struct ImuFactor {
  ImuFactor(const ImuState& i1, const ImuState& i2, const ImuState& i3,
            double sp1_timestamp, double sp2_timestamp, double sp3_timestamp,
            double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
            double dt, const Vector3d& gravity) : i1_(i1), i2_(i2), i3_(i3), sp1_timestamp_(sp1_timestamp), sp2_timestamp_(sp2_timestamp), sp3_timestamp_(sp3_timestamp), weight_gyr_(weight_gyr), weight_acc_(weight_acc), weight_bg_(weight_bg), weight_ba_(weight_ba), dt_(dt), gravity_(gravity) {
  }

  template <typename T>
  bool operator()(const T* sp1_ptr, const T* sp2_ptr, const T* sp3_ptr, T* residuals) const {
    Eigen::Map<const Vector12<T>> sp1{sp1_ptr};
    Eigen::Map<const Vector12<T>> sp2{sp2_ptr};
    Eigen::Map<const Vector12<T>> sp3{sp3_ptr};
    Eigen::Matrix<T, 3, 1>        r_i1_cor, t_i1_cor, bg_i1, ba_i1;
    Eigen::Matrix<T, 3, 1>        r_i2_cor, t_i2_cor, bg_i2, ba_i2;
    Eigen::Matrix<T, 3, 1>        r_i3_cor, t_i3_cor, bg_i3, ba_i3;
    ComputeStateCorr(sp1, sp2, sp3, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, i1_.timestamp, r_i1_cor, t_i1_cor, bg_i1, ba_i1);
    ComputeStateCorr(sp1, sp2, sp3, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, i2_.timestamp, r_i2_cor, t_i2_cor, bg_i2, ba_i2);
    ComputeStateCorr(sp1, sp2, sp3, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, i3_.timestamp, r_i3_cor, t_i3_cor, bg_i3, ba_i3);
    Helper(r_i1_cor, t_i1_cor, bg_i1, ba_i1, r_i2_cor, t_i2_cor, bg_i2, ba_i2, r_i3_cor, t_i3_cor, bg_i3, ba_i3, residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* sp1_ptr, const T* sp2_ptr, T* residuals) const {
    Eigen::Map<const Vector12<T>> sp1{sp1_ptr};
    Eigen::Map<const Vector12<T>> sp2{sp2_ptr};
    Eigen::Matrix<T, 3, 1>        r_i1_cor, t_i1_cor, bg_i1, ba_i1;
    Eigen::Matrix<T, 3, 1>        r_i2_cor, t_i2_cor, bg_i2, ba_i2;
    Eigen::Matrix<T, 3, 1>        r_i3_cor, t_i3_cor, bg_i3, ba_i3;
    ComputeStateCorr(sp1, sp2, sp1_timestamp_, sp2_timestamp_, i1_.timestamp, r_i1_cor, t_i1_cor, bg_i1, ba_i1);
    ComputeStateCorr(sp1, sp2, sp1_timestamp_, sp2_timestamp_, i2_.timestamp, r_i2_cor, t_i2_cor, bg_i2, ba_i2);
    ComputeStateCorr(sp1, sp2, sp1_timestamp_, sp2_timestamp_, i3_.timestamp, r_i3_cor, t_i3_cor, bg_i3, ba_i3);
    Helper(r_i1_cor, t_i1_cor, bg_i1, ba_i1, r_i2_cor, t_i2_cor, bg_i2, ba_i2, r_i3_cor, t_i3_cor, bg_i3, ba_i3, residuals);
    return true;
  }

  static ceres::CostFunction* Create(const ImuState& i1, const ImuState& i2, const ImuState& i3,
                                     double sp1_timestamp, double sp2_timestamp, double sp3_timestamp,
                                     double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
                                     double dt, const Vector3d& gravity) {
    if constexpr (Mode == 0) {
      return (new ceres::AutoDiffCostFunction<ImuFactor, 12, 12, 12, 12>(
          new ImuFactor(i1, i2, i3, sp1_timestamp, sp2_timestamp, sp3_timestamp, weight_gyr, weight_acc, weight_bg, weight_ba, dt, gravity)));
    } else {
      return (new ceres::AutoDiffCostFunction<ImuFactor, 12, 12, 12>(
          new ImuFactor(i1, i2, i3, sp1_timestamp, sp2_timestamp, sp3_timestamp, weight_gyr, weight_acc, weight_bg, weight_ba, dt, gravity)));
    }
  }

 private:
  template <typename T>
  void Helper(
      Eigen::Matrix<T, 3, 1> r_i1_cor, Eigen::Matrix<T, 3, 1> t_i1_cor, Eigen::Matrix<T, 3, 1> bg_i1, Eigen::Matrix<T, 3, 1> ba_i1,
      Eigen::Matrix<T, 3, 1> r_i2_cor, Eigen::Matrix<T, 3, 1> t_i2_cor, Eigen::Matrix<T, 3, 1> bg_i2, Eigen::Matrix<T, 3, 1> ba_i2,
      Eigen::Matrix<T, 3, 1> r_i3_cor, Eigen::Matrix<T, 3, 1> t_i3_cor, Eigen::Matrix<T, 3, 1> bg_i3, Eigen::Matrix<T, 3, 1> ba_i3,
      T* residuals) const {
    Eigen::Matrix<T, 3, 1> gyr_est = Log((Exp(r_i1_cor) * i1_.rot.cast<T>()).conjugate() * Exp(r_i2_cor) * i2_.rot.cast<T>()) / dt_;
    Eigen::Matrix<T, 3, 1> acc_est = ((t_i3_cor + i3_.pos.cast<T>()) + (t_i1_cor + i1_.pos.cast<T>()) - 2.0 * (t_i2_cor + i2_.pos.cast<T>())) / (dt_ * dt_);

    Eigen::Map<Vector12<T>> r{residuals};
    r.template block<3, 1>(0, 0) = weight_gyr_ * ((i1_.gyr.cast<T>() + i2_.gyr.cast<T>()) / 2.0 - gyr_est - bg_i1);
    r.template block<3, 1>(3, 0) = weight_acc_ * ((Exp(r_i1_cor) * i1_.rot.cast<T>()) * (i1_.acc.cast<T>() - ba_i1) - acc_est + gravity_.cast<T>());
    r.template block<3, 1>(6, 0) = weight_bg_ * (bg_i1 - bg_i2);
    r.template block<3, 1>(9, 0) = weight_ba_ * (ba_i1 - ba_i2);
  }

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
      Eigen::Matrix<T, 3, 1>&        ba) const {
    CHECK((timestamp >= sp1_timestamp && timestamp < sp2_timestamp) || (timestamp >= sp2_timestamp && timestamp <= sp3_timestamp))
        << std::fixed << std::setprecision(6) << "timestamp: " << timestamp << " sp1: " << sp1_timestamp << " sp2: " << sp2_timestamp << " sp3: " << sp3_timestamp;

    bool between_sp1_sp2 = (timestamp >= sp1_timestamp && timestamp < sp2_timestamp);

    const Eigen::Map<const Vector12<T>>& spl           = between_sp1_sp2 ? sp1 : sp2;
    const Eigen::Map<const Vector12<T>>& spr           = between_sp1_sp2 ? sp2 : sp3;
    double                               spl_timestamp = between_sp1_sp2 ? sp1_timestamp : sp2_timestamp;
    double                               spr_timestamp = between_sp1_sp2 ? sp2_timestamp : sp3_timestamp;

    double      factor = (timestamp - spl_timestamp) / (spr_timestamp - spl_timestamp);
    Vector12<T> sp     = (1 - factor) * spl + factor * spr;
    r_cor              = sp.template block<3, 1>(0, 0);
    t_cor              = sp.template block<3, 1>(3, 0);
    bg                 = sp.template block<3, 1>(6, 0);
    ba                 = sp.template block<3, 1>(9, 0);
  }

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
      Eigen::Matrix<T, 3, 1>&              ba) const {
    CHECK(timestamp >= sp1_timestamp && timestamp <= sp2_timestamp) << std::fixed << std::setprecision(6) << "Timestamp order: " << timestamp << " sp1: " << sp1_timestamp << " sp2: " << sp2_timestamp;

    double      factor = (timestamp - sp1_timestamp) / (sp2_timestamp - sp1_timestamp);
    Vector12<T> sp     = (1 - factor) * sp1 + factor * sp2;
    r_cor              = sp.template block<3, 1>(0, 0);
    t_cor              = sp.template block<3, 1>(3, 0);
    bg                 = sp.template block<3, 1>(6, 0);
    ba                 = sp.template block<3, 1>(9, 0);
  }

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
