#pragma once

#include <ceres/ceres.h>
#include <iomanip>

#include "common/utils.h"
#include "odometry/surfel.h"

/**
 * @brief s1 s2 is two corresponding surfels
 *
 * 1. Timestamp order: s1 < sp1l <= s1 < sp1r
 * 2. s1 s2 are not in adjacent sampled intervals
 *
 */
struct SurfelMatchUnaryFactor : public ceres::SizedCostFunction<1, 12, 12> {
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

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> r_sp2l{&parameters[0][0]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_sp2l{&parameters[0][3]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> r_sp2r{&parameters[1][0]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_sp2r{&parameters[1][3]};
    double                                        factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
    Eigen::Matrix<double, 3, 1>                   r_s2    = (1 - factor2) * r_sp2l + factor2 * r_sp2r;
    Eigen::Matrix<double, 3, 1>                   t_s2    = (1 - factor2) * t_sp2l + factor2 * t_sp2r;
    CHECK_GE(factor2, 0);
    CHECK_LE(factor2, 1);

    residuals[0] = weight_ * norm_.dot(s1_->CenterInBody() - Exp(r_s2) * s2_->rot * s2_->CenterInBody() - s2_->pos - t_s2);

    if (jacobians) {
      Eigen::Matrix<double, 1, 12> jacobian_s2;
      jacobian_s2.setZero();
      jacobian_s2.block<1, 3>(0, 0) = weight_ * norm_.transpose() * Exp(r_s2).matrix() * Hat(s2_->rot * s2_->CenterInBody()) * Jr(r_s2);
      jacobian_s2.block<1, 3>(0, 3) = -weight_ * norm_.transpose();

      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>> jacobian_sp2l{jacobians[2]};
        jacobian_sp2l = jacobian_s2 * (1 - factor2);
      }

      if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>> jacobian_sp2r{jacobians[3]};
        jacobian_sp2r = jacobian_s2 * factor2;
      }
    }

    return true;
  }

 private:
  std::shared_ptr<Surfel>      s1_;
  std::shared_ptr<SampleState> sp2l_;
  std::shared_ptr<SampleState> sp2r_;
  std::shared_ptr<Surfel>      s2_;

  Vector3d norm_;
  double   weight_;
};

template <int Mode>
struct SurfelMatchBinaryModeTraits {
};

template <>
struct SurfelMatchBinaryModeTraits<0> {
  using type = ceres::SizedCostFunction<1, 12, 12, 12, 12>;
};

template <>
struct SurfelMatchBinaryModeTraits<1> {
  using type = ceres::SizedCostFunction<1, 12, 12, 12>;
};

template <>
struct SurfelMatchBinaryModeTraits<2> {
  using type = ceres::SizedCostFunction<1, 12, 12>;
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
template <int Mode, typename TMode = typename SurfelMatchBinaryModeTraits<Mode>::type>
struct SurfelMatchBinaryFactor : public TMode {
  SurfelMatchBinaryFactor(
      std::shared_ptr<Surfel>      s1,
      std::shared_ptr<SampleState> sp1l,
      std::shared_ptr<SampleState> sp1r,
      std::shared_ptr<Surfel>      s2,
      std::shared_ptr<SampleState> sp2l,
      std::shared_ptr<SampleState> sp2r) : s1_(s1), sp1l_(sp1l), sp1r_(sp1r), s2_(s2), sp2l_(sp2l), sp2r_(sp2r) {
    ValidateTimestamps();
    Matrix3d                                cov = s1_->GetCovarianceInWorld() + s2_->GetCovarianceInWorld();
    Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
    weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
    norm_   = es.eigenvectors().col(0);
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    const double *sp1l_ptr, *sp1r_ptr, *sp2l_ptr, *sp2r_ptr;
    DispatchPtr(parameters, sp1l_ptr, sp1r_ptr, sp2l_ptr, sp2r_ptr);

    Eigen::Map<const Eigen::Matrix<double, 3, 1>> r_sp1l{&sp1l_ptr[0]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_sp1l{&sp1l_ptr[3]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> r_sp1r{&sp1r_ptr[0]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_sp1r{&sp1r_ptr[3]};
    double                                        factor1 = (s1_->timestamp - sp1l_->timestamp) / (sp1r_->timestamp - sp1l_->timestamp);
    Eigen::Matrix<double, 3, 1>                   r_s1    = (1 - factor1) * r_sp1l + factor1 * r_sp1r;
    Eigen::Matrix<double, 3, 1>                   t_s1    = (1 - factor1) * t_sp1l + factor1 * t_sp1r;
    CHECK_GE(factor1, 0);
    CHECK_LE(factor1, 1);

    Eigen::Map<const Eigen::Matrix<double, 3, 1>> r_sp2l{&sp2l_ptr[0]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_sp2l{&sp2l_ptr[3]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> r_sp2r{&sp2r_ptr[0]};
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_sp2r{&sp2r_ptr[3]};
    double                                        factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
    Eigen::Matrix<double, 3, 1>                   r_s2    = (1 - factor2) * r_sp2l + factor2 * r_sp2r;
    Eigen::Matrix<double, 3, 1>                   t_s2    = (1 - factor2) * t_sp2l + factor2 * t_sp2r;
    CHECK_GE(factor2, 0);
    CHECK_LE(factor2, 1);

    residuals[0] = weight_ * norm_.dot(Exp(r_s1) * s1_->rot * s1_->CenterInBody() + t_s1 + s1_->pos - Exp(r_s2) * s2_->rot * s2_->CenterInBody() - t_s2 - s2_->pos);

    if (jacobians) {
      InitJacobians(jacobians);
      double *sp1l_jacobian_ptr, *sp1r_jacobian_ptr, *sp2l_jacobian_ptr, *sp2r_jacobian_ptr;
      DispatchPtr(jacobians, sp1l_jacobian_ptr, sp1r_jacobian_ptr, sp2l_jacobian_ptr, sp2r_jacobian_ptr);

      Eigen::Matrix<double, 1, 12> jacobian_s1;
      jacobian_s1.setZero();
      jacobian_s1.block<1, 3>(0, 0) = -weight_ * norm_.transpose() * Exp(r_s1).matrix() * Hat(s1_->rot * s1_->CenterInBody()) * Jr(r_s1);
      jacobian_s1.block<1, 3>(0, 3) = weight_ * norm_.transpose();

      if (sp1l_jacobian_ptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>> jacobian_sp1l{sp1l_jacobian_ptr};
        jacobian_sp1l = jacobian_s1 * (1 - factor1);
      }

      if (sp1r_jacobian_ptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>> jacobian_sp1r{sp1r_jacobian_ptr};
        jacobian_sp1r = jacobian_s1 * factor1;
      }

      Eigen::Matrix<double, 1, 12> jacobian_s2;
      jacobian_s2.setZero();
      jacobian_s2.block<1, 3>(0, 0) = weight_ * norm_.transpose() * Exp(r_s2).matrix() * Hat(s2_->rot * s2_->CenterInBody()) * Jr(r_s2);
      jacobian_s2.block<1, 3>(0, 3) = -weight_ * norm_.transpose();

      if (sp2l_jacobian_ptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>> jacobian_sp2l{sp2l_jacobian_ptr};
        jacobian_sp2l = jacobian_s2 * (1 - factor2);
      }

      if (sp2r_jacobian_ptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>> jacobian_sp2r{sp2r_jacobian_ptr};
        jacobian_sp2r = jacobian_s2 * factor2;
      }
    }

    return true;
  }

 private:
  void ValidateTimestamps() const {
    CHECK_LT(s1_->timestamp, s2_->timestamp);
    if constexpr (Mode == 0) {
      CHECK_LT(sp1r_->timestamp, sp2l_->timestamp);
    } else if constexpr (Mode == 1) {
      CHECK_EQ(sp1r_->timestamp, sp2l_->timestamp);
    } else {
      CHECK_EQ(sp1l_->timestamp, sp2l_->timestamp);
      CHECK_EQ(sp1r_->timestamp, sp2r_->timestamp);
    }
  }

  void InitJacobians(double** jacobians) const {
    if (!jacobians) {
      return;
    }

#define SET_JACOBIAN_TO_ZERO(dim)                                                        \
  if (!jacobians[dim]) {                                                                 \
    Eigen::Map<Eigen::Matrix<double, 1, 12, Eigen::RowMajor>>{jacobians[dim]}.setZero(); \
  }

    SET_JACOBIAN_TO_ZERO(0);
    SET_JACOBIAN_TO_ZERO(1);
    if constexpr (Mode == 0) {
      SET_JACOBIAN_TO_ZERO(2);
      SET_JACOBIAN_TO_ZERO(3);
    } else if constexpr (Mode == 1) {
      SET_JACOBIAN_TO_ZERO(2);
    } else {
    }
  }

  template <typename T>
  void DispatchPtr(T* const* ptrs, T*& sp1l_ptr, T*& sp1r_ptr, T*& sp2l_ptr, T*& sp2r_ptr) const {
    sp1l_ptr = ptrs[0];
    sp1r_ptr = ptrs[1];
    if (Mode == 0) {
      sp2l_ptr = ptrs[2];
      sp2r_ptr = ptrs[3];
    } else if (Mode == 1) {
      sp2l_ptr = ptrs[1];
      sp2r_ptr = ptrs[2];
    } else {
      sp2l_ptr = ptrs[0];
      sp2r_ptr = ptrs[1];
    }
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
struct ImuFactor : public TMode {
  ImuFactor(const ImuState& i1, const ImuState& i2, const ImuState& i3,
            double sp1_timestamp, double sp2_timestamp, double sp3_timestamp,
            double weight_gyr, double weight_acc, double weight_bg, double weight_ba,
            double dt, const Vector3d& gravity) : i1_(i1), i2_(i2), i3_(i3), sp1_timestamp_(sp1_timestamp), sp2_timestamp_(sp2_timestamp), sp3_timestamp_(sp3_timestamp), weight_gyr_(weight_gyr), weight_acc_(weight_acc), weight_bg_(weight_bg), weight_ba_(weight_ba), dt_(dt), gravity_(gravity) {
  }

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    SampleState sp1 = ToSampleState(parameters[0], sp1_timestamp_);
    SampleState sp2 = ToSampleState(parameters[1], sp2_timestamp_);

    Vector3d r_i1_cor, t_i1_cor, bg_i1, ba_i1;
    Vector3d r_i2_cor, t_i2_cor, bg_i2, ba_i2;
    Vector3d r_i3_cor, t_i3_cor, bg_i3, ba_i3;

    if constexpr (Mode == 0) {
      SampleState sp3 = ToSampleState(parameters[2], sp3_timestamp_);
      ComputeStateCorr(sp1, sp2, sp3, i1_.timestamp, r_i1_cor, t_i1_cor, bg_i1, ba_i1);
      ComputeStateCorr(sp1, sp2, sp3, i2_.timestamp, r_i2_cor, t_i2_cor, bg_i2, ba_i2);
      ComputeStateCorr(sp1, sp2, sp3, i3_.timestamp, r_i3_cor, t_i3_cor, bg_i3, ba_i3);
    } else {
      ComputeStateCorr(sp1, sp2, i1_.timestamp, r_i1_cor, t_i1_cor, bg_i1, ba_i1);
      ComputeStateCorr(sp1, sp2, i2_.timestamp, r_i2_cor, t_i2_cor, bg_i2, ba_i2);
      ComputeStateCorr(sp1, sp2, i3_.timestamp, r_i3_cor, t_i3_cor, bg_i3, ba_i3);
    }

    Vector3d gyr_est = Log((Exp(r_i1_cor) * i1_.rot).conjugate() * Exp(r_i2_cor) * i2_.rot) / dt_;
    Vector3d acc_est = ((t_i3_cor + i3_.pos) + (t_i1_cor + i1_.pos) - 2 * (t_i2_cor + i2_.pos)) / (dt_ * dt_);

    Eigen::Map<Eigen::Matrix<double, 12, 1>> r{residuals};
    r.block<3, 1>(0, 0) = weight_gyr_ * ((i1_.gyr + i2_.gyr) / 2 - gyr_est - bg_i1);
    r.block<3, 1>(3, 0) = weight_acc_ * ((Exp(r_i1_cor) * i1_.rot) * (i1_.acc - ba_i1) - acc_est + gravity_);
    r.block<3, 1>(6, 0) = weight_bg_ * (bg_i1 - bg_i2);
    r.block<3, 1>(9, 0) = weight_ba_ * (ba_i1 - ba_i2);

    if (jacobians) {
      Eigen::Matrix<double, 12, 12> jacobian_tau;
      jacobian_tau.setZero();
      jacobian_tau.block<3, 3>(0, 0) = weight_gyr_ * (1 / dt_) * F(i1_.rot.conjugate(), Exp(r_i2_cor) * i2_.rot, r_i1_cor);
      jacobian_tau.block<3, 3>(0, 6) = -weight_gyr_ * Matrix3d::Identity();
      jacobian_tau.block<3, 3>(3, 0) = -weight_acc_ * (Exp(r_i1_cor).matrix() * Hat(i1_.rot * (i1_.acc - ba_i1)) * Jr(r_i1_cor));
      jacobian_tau.block<3, 3>(3, 3) = -weight_acc_ * (1 / dt_ / dt_) * Matrix3d::Identity();
      jacobian_tau.block<3, 3>(3, 9) = -weight_acc_ * (Exp(r_i1_cor) * i1_.rot).matrix();
      jacobian_tau.block<3, 3>(6, 6) = weight_bg_ * Matrix3d::Identity();
      jacobian_tau.block<3, 3>(9, 9) = weight_ba_ * Matrix3d::Identity();

      Eigen::Matrix<double, 12, 12> jacobian_tau1;
      jacobian_tau1.setZero();
      jacobian_tau1.block<3, 3>(0, 0) = -weight_gyr_ * (1 / dt_) * F((Exp(r_i1_cor) * i1_.rot).conjugate(), i2_.rot, r_i2_cor);
      jacobian_tau1.block<3, 3>(0, 6) = -weight_gyr_ * Matrix3d::Identity();
      jacobian_tau1.block<3, 3>(3, 3) = weight_acc_ * (2 / dt_ / dt_) * Matrix3d::Identity();
      jacobian_tau1.block<3, 3>(6, 6) = -weight_bg_ * Matrix3d::Identity();
      jacobian_tau1.block<3, 3>(9, 9) = -weight_ba_ * Matrix3d::Identity();

      Eigen::Matrix<double, 12, 12> jacobian_tau2;
      jacobian_tau2.setZero();
      jacobian_tau2.block<3, 3>(3, 3) = -weight_acc_ * (1 / dt_ / dt_) * Matrix3d::Identity();

      if constexpr (Mode == 0) {
        if (!jacobians[0] || !jacobians[1] || !jacobians[2]) {
          LOG(FATAL) << "unimplemented";
        }

        Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp1{jacobians[0]};
        Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp2{jacobians[1]};
        Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp3{jacobians[2]};
        jacobian_sp1.setZero();
        jacobian_sp2.setZero();
        jacobian_sp3.setZero();

        DispatchJacobians(jacobian_tau, i1_.timestamp, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, jacobian_sp1, jacobian_sp2, jacobian_sp3);
        DispatchJacobians(jacobian_tau1, i2_.timestamp, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, jacobian_sp1, jacobian_sp2, jacobian_sp3);
        DispatchJacobians(jacobian_tau2, i3_.timestamp, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, jacobian_sp1, jacobian_sp2, jacobian_sp3);
      } else {
        if (!jacobians[0] || !jacobians[1]) {
          LOG(FATAL) << "unimplemented";
        }

        Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp1{jacobians[0]};
        Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp2{jacobians[1]};
        jacobian_sp1.setZero();
        jacobian_sp2.setZero();

        DispatchJacobians(jacobian_tau, i1_.timestamp, sp1_timestamp_, sp2_timestamp_, jacobian_sp1, jacobian_sp2);
        DispatchJacobians(jacobian_tau1, i2_.timestamp, sp1_timestamp_, sp2_timestamp_, jacobian_sp1, jacobian_sp2);
        DispatchJacobians(jacobian_tau2, i3_.timestamp, sp1_timestamp_, sp2_timestamp_, jacobian_sp1, jacobian_sp2);
      }
    }

    return true;
  }

 private:
  void ComputeStateCorr(
      const SampleState& sp1,
      const SampleState& sp2,
      const SampleState& sp3,
      const double&      timestamp,
      Vector3d&          r_cor,
      Vector3d&          t_cor,
      Vector3d&          bg,
      Vector3d&          ba) const {
    CHECK((timestamp >= sp1.timestamp && timestamp < sp2.timestamp) || (timestamp >= sp2.timestamp && timestamp <= sp3.timestamp))
        << std::fixed << std::setprecision(6) << "timestamp: " << timestamp << " sp1: " << sp1.timestamp << " sp2: " << sp2.timestamp << " sp3: " << sp3.timestamp;

    bool between_sp1_sp2 = (timestamp >= sp1.timestamp && timestamp < sp2.timestamp);

    const SampleState& spl = between_sp1_sp2 ? sp1 : sp2;
    const SampleState& spr = between_sp1_sp2 ? sp2 : sp3;

    double factor = (timestamp - spl.timestamp) / (spr.timestamp - spl.timestamp);
    r_cor         = (1 - factor) * spl.rot_cor + factor * spr.rot_cor;
    t_cor         = (1 - factor) * spl.pos_cor + factor * spr.pos_cor;
    bg            = (1 - factor) * spl.bg + factor * spr.bg;
    ba            = (1 - factor) * spl.ba + factor * spr.ba;
  }

  void ComputeStateCorr(
      const SampleState& sp1,
      const SampleState& sp2,
      const double&      timestamp,
      Vector3d&          r_cor,
      Vector3d&          t_cor,
      Vector3d&          bg,
      Vector3d&          ba) const {
    CHECK(timestamp >= sp1.timestamp && timestamp <= sp2.timestamp) << std::fixed << std::setprecision(6) << "Timestamp order: " << timestamp << " sp1: " << sp1.timestamp << " sp2: " << sp2.timestamp;

    const SampleState& spl = sp1;
    const SampleState& spr = sp2;

    double factor = (timestamp - spl.timestamp) / (spr.timestamp - spl.timestamp);
    r_cor         = (1 - factor) * spl.rot_cor + factor * spr.rot_cor;
    t_cor         = (1 - factor) * spl.pos_cor + factor * spr.pos_cor;
    bg            = (1 - factor) * spl.bg + factor * spr.bg;
    ba            = (1 - factor) * spl.ba + factor * spr.ba;
  }

  void DispatchJacobians(
      const Eigen::Matrix<double, 12, 12>&                       jacobian_tau,
      double                                                     timestamp,
      double                                                     timestamp_sp1,
      double                                                     timestamp_sp2,
      double                                                     timestamp_sp3,
      Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp1,
      Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp2,
      Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp3) const {
    CHECK((timestamp >= timestamp_sp1 && timestamp < timestamp_sp2) || (timestamp >= timestamp_sp2 && timestamp <= timestamp_sp3));

    bool between_sp1_sp2 = (timestamp >= timestamp_sp1 && timestamp < timestamp_sp2);

    auto timestamp_spl = between_sp1_sp2 ? timestamp_sp1 : timestamp_sp2;
    auto timestamp_spr = between_sp1_sp2 ? timestamp_sp2 : timestamp_sp3;
    auto jl            = between_sp1_sp2 ? jacobian_sp1 : jacobian_sp2;
    auto jr            = between_sp1_sp2 ? jacobian_sp2 : jacobian_sp3;

    double factor = (timestamp - timestamp_spl) / (timestamp_spr - timestamp_spl);

    jl += jacobian_tau * (1 - factor);
    jr += jacobian_tau * factor;
  }

  void DispatchJacobians(
      const Eigen::Matrix<double, 12, 12>&                       jacobian_tau,
      double                                                     timestamp,
      double                                                     timestamp_sp1,
      double                                                     timestamp_sp2,
      Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp1,
      Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::RowMajor>> jacobian_sp2) const {
    CHECK(timestamp >= timestamp_sp1 && timestamp <= timestamp_sp2);

    auto timestamp_spl = timestamp_sp1;
    auto timestamp_spr = timestamp_sp2;
    auto jl            = jacobian_sp1;
    auto jr            = jacobian_sp2;

    double factor = (timestamp - timestamp_spl) / (timestamp_spr - timestamp_spl);

    jl += jacobian_tau * (1 - factor);
    jr += jacobian_tau * factor;
  }

  Matrix3d F(const Quaterniond& L, const Quaterniond& R, const Vector3d& r) const {
    return Jr_inv(Log(L * Exp(r) * R)) * R.conjugate().matrix() * Jr(r);
  }

  SampleState ToSampleState(const double* const parameters, double timestamp) const {
    SampleState sp;
    sp.timestamp = timestamp;
    sp.rot_cor   = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(parameters);
    sp.pos_cor   = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(parameters + 3);
    sp.bg        = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(parameters + 6);
    sp.ba        = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(parameters + 9);
    return sp;
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
