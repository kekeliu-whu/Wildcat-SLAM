#include <iomanip>

#include "common/utils.h"
#include "cost_functor.h"
#include "symforce/sym/surfel_binary_match_2_samples_with_jacobians01.h"
#include "symforce/sym/surfel_binary_match_3_samples_with_jacobians012.h"
#include "symforce/sym/surfel_binary_match_4_samples_with_jacobians0123.h"
#include "symforce/sym/surfel_unary_match_with_jacobians01.h"

static constexpr double eps = std::numeric_limits<float>::epsilon();

SurfelMatchUnaryFactor::SurfelMatchUnaryFactor(
    std::shared_ptr<Surfel>      s1,
    std::shared_ptr<Surfel>      s2,
    std::shared_ptr<SampleState> sp2l,
    std::shared_ptr<SampleState> sp2r) : s1_(s1), sp2l_(sp2l), sp2r_(sp2r), s2_(s2) {
  Matrix3d                                cov = s1_->covariance + s2_->covariance;
  Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
  weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
  norm_   = es.eigenvectors().col(0);
}

bool SurfelMatchUnaryFactor::Evaluate(double const *const *parameters,
                                      double              *residuals,
                                      double             **jacobians) const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp2l_cor{parameters[0]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp2r_cor{parameters[1]};

  double factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
  DCHECK_GE(factor2, 0.0);
  DCHECK_LE(factor2, 1.0);

  residuals[0] = sym::SurfelUnaryMatchWithJacobians01(
      sp2l_cor, sp2r_cor, factor2, weight_, norm_, s1_->center, s2_->center,
      eps,
      jacobians ? jacobians[0] : nullptr,
      jacobians ? jacobians[1] : nullptr);

  return true;
}

template <typename T>
void ImuFactorHelper(
    Eigen::Quaternion<T> &R_i1_cor, Eigen::Matrix<T, 3, 1> &t_i1_cor,
    Eigen::Quaternion<T> &R_i2_cor, Eigen::Matrix<T, 3, 1> &t_i2_cor,
    Eigen::Quaternion<T> &R_i3_cor, Eigen::Matrix<T, 3, 1> &t_i3_cor,
    Eigen::Matrix<T, 3, 1> &bg, Eigen::Matrix<T, 3, 1> ba,
    const ImuState &i1_,
    const ImuState &i2_,
    const ImuState &i3_,
    const Vector3d &gravity_,
    double          dt_,
    double weight_gyr_, double weight_acc_, double weight_bg_, double weight_ba_,
    T *residuals) {
  Eigen::Matrix<T, 3, 1> gyr_est = Log((R_i1_cor * i1_.rot.cast<T>()).conjugate() * R_i2_cor * i2_.rot.cast<T>()) / dt_;
  Eigen::Matrix<T, 3, 1> acc_est = ((t_i3_cor + R_i3_cor * i3_.pos.cast<T>()) + (t_i1_cor + R_i1_cor * i1_.pos.cast<T>()) - 2.0 * (t_i2_cor + R_i2_cor * i2_.pos.cast<T>())) / (dt_ * dt_);

  Eigen::Map<Vector6<T>> r{residuals};
  r.template block<3, 1>(0, 0) = weight_gyr_ * ((i1_.gyr.cast<T>() + i2_.gyr.cast<T>()) / 2.0 - gyr_est - bg);
  r.template block<3, 1>(3, 0) = weight_acc_ * ((R_i1_cor * i1_.rot.cast<T>()) * (i1_.acc.cast<T>() - ba) - acc_est + gravity_.cast<T>());
}

ImuFactorWith3SampleStates::ImuFactorWith3SampleStates(const ImuState &i1, const ImuState &i2, const ImuState &i3, double sp1_timestamp, double sp2_timestamp, double sp3_timestamp, double weight_gyr, double weight_acc, double weight_bg, double weight_ba, double dt, const Vector3d &gravity) : i1_(i1), i2_(i2), i3_(i3), sp1_timestamp_(sp1_timestamp), sp2_timestamp_(sp2_timestamp), sp3_timestamp_(sp3_timestamp), weight_gyr_(weight_gyr), weight_acc_(weight_acc), weight_bg_(weight_bg), weight_ba_(weight_ba), dt_(dt), gravity_(gravity) {
}

template <typename T>
bool ImuFactorWith3SampleStates::operator()(const T *sp1_ptr, const T *sp2_ptr, const T *sp3_ptr, const T *bias_ptr, T *residuals) const {
  Eigen::Map<const Vector6<T>> sp1{sp1_ptr};
  Eigen::Map<const Vector6<T>> sp2{sp2_ptr};
  Eigen::Map<const Vector6<T>> sp3{sp3_ptr};
  Eigen::Quaternion<T>         R_i1_cor, R_i2_cor, R_i3_cor;
  Eigen::Matrix<T, 3, 1>       t_i1_cor, t_i2_cor, t_i3_cor;
  Eigen::Matrix<T, 3, 1>       bg = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(bias_ptr + 0);
  Eigen::Matrix<T, 3, 1>       ba = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(bias_ptr + 3);
  ComputeStateCorr(sp1, sp2, sp3, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, i1_.timestamp, R_i1_cor, t_i1_cor);
  ComputeStateCorr(sp1, sp2, sp3, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, i2_.timestamp, R_i2_cor, t_i2_cor);
  ComputeStateCorr(sp1, sp2, sp3, sp1_timestamp_, sp2_timestamp_, sp3_timestamp_, i3_.timestamp, R_i3_cor, t_i3_cor);
  ImuFactorHelper(R_i1_cor, t_i1_cor, R_i2_cor, t_i2_cor, R_i3_cor, t_i3_cor, bg, ba, i1_, i2_, i3_, gravity_, dt_, weight_gyr_, weight_acc_, weight_bg_, weight_ba_, residuals);
  return true;
}

ceres::CostFunction *ImuFactorWith3SampleStates::Create(const ImuState &i1, const ImuState &i2, const ImuState &i3, double sp1_timestamp, double sp2_timestamp, double sp3_timestamp, double weight_gyr, double weight_acc, double weight_bg, double weight_ba, double dt, const Vector3d &gravity) {
  return (new ceres::AutoDiffCostFunction<ImuFactorWith3SampleStates, 6, 6, 6, 6, 6>(
      new ImuFactorWith3SampleStates(i1, i2, i3, sp1_timestamp, sp2_timestamp, sp3_timestamp, weight_gyr, weight_acc, weight_bg, weight_ba, dt, gravity)));
}

template <typename T>
void ImuFactorWith3SampleStates::ComputeStateCorr(Eigen::Map<const Vector6<T>> &sp1, Eigen::Map<const Vector6<T>> &sp2, Eigen::Map<const Vector6<T>> &sp3, double sp1_timestamp, double sp2_timestamp, double sp3_timestamp, double timestamp, Eigen::Quaternion<T> &R_cor, Eigen::Matrix<T, 3, 1> &t_cor) const {
  CHECK((timestamp >= sp1_timestamp && timestamp < sp2_timestamp) || (timestamp >= sp2_timestamp && timestamp <= sp3_timestamp))
      << std::fixed << std::setprecision(9) << "timestamp: " << timestamp << " sp1: " << sp1_timestamp << " sp2: " << sp2_timestamp << " sp3: " << sp3_timestamp;

  bool between_sp1_sp2 = (timestamp >= sp1_timestamp && timestamp < sp2_timestamp);

  const Eigen::Map<const Vector6<T>> &spl           = between_sp1_sp2 ? sp1 : sp2;
  const Eigen::Map<const Vector6<T>> &spr           = between_sp1_sp2 ? sp2 : sp3;
  double                              spl_timestamp = between_sp1_sp2 ? sp1_timestamp : sp2_timestamp;
  double                              spr_timestamp = between_sp1_sp2 ? sp2_timestamp : sp3_timestamp;

  double factor = (timestamp - spl_timestamp) / (spr_timestamp - spl_timestamp);
  // interpolate rot
  Sophus::SO3<T> spl_rot = Sophus::SO3<T>::exp(spl.template head<3>());
  Sophus::SO3<T> spr_rot = Sophus::SO3<T>::exp(spr.template head<3>());
  R_cor                  = (spl_rot * Sophus::SO3<T>::exp(factor * (spl_rot.inverse() * spr_rot).log())).unit_quaternion();
  // interpolate pos and bias
  Vector6<T> sp = (1 - factor) * spl + factor * spr;
  t_cor         = sp.template block<3, 1>(3, 0);
}

ImuFactorWith2SampleStates::ImuFactorWith2SampleStates(const ImuState &i1, const ImuState &i2, const ImuState &i3, double sp1_timestamp, double sp2_timestamp, double weight_gyr, double weight_acc, double weight_bg, double weight_ba, double dt, const Vector3d &gravity) : i1_(i1), i2_(i2), i3_(i3), sp1_timestamp_(sp1_timestamp), sp2_timestamp_(sp2_timestamp), weight_gyr_(weight_gyr), weight_acc_(weight_acc), weight_bg_(weight_bg), weight_ba_(weight_ba), dt_(dt), gravity_(gravity) {
}

template <typename T>
bool ImuFactorWith2SampleStates::operator()(const T *sp1_ptr, const T *sp2_ptr, const T *bias_ptr, T *residuals) const {
  Eigen::Map<const Vector6<T>> sp1{sp1_ptr};
  Eigen::Map<const Vector6<T>> sp2{sp2_ptr};

  Eigen::Quaternion<T>   R_i1_cor, R_i2_cor, R_i3_cor;
  Eigen::Matrix<T, 3, 1> t_i1_cor, t_i2_cor, t_i3_cor;
  Eigen::Matrix<T, 3, 1> bg = Eigen::Map<const Eigen::Matrix<T, 3, 1>>{bias_ptr + 0};
  Eigen::Matrix<T, 3, 1> ba = Eigen::Map<const Eigen::Matrix<T, 3, 1>>{bias_ptr + 3};
  ComputeStateCorr(sp1, sp2, sp1_timestamp_, sp2_timestamp_, i1_.timestamp, R_i1_cor, t_i1_cor);
  ComputeStateCorr(sp1, sp2, sp1_timestamp_, sp2_timestamp_, i2_.timestamp, R_i2_cor, t_i2_cor);
  ComputeStateCorr(sp1, sp2, sp1_timestamp_, sp2_timestamp_, i3_.timestamp, R_i3_cor, t_i3_cor);
  ImuFactorHelper(R_i1_cor, t_i1_cor, R_i2_cor, t_i2_cor, R_i3_cor, t_i3_cor, bg, ba, i1_, i2_, i3_, gravity_, dt_, weight_gyr_, weight_acc_, weight_bg_, weight_ba_, residuals);
  return true;
}

ceres::CostFunction *ImuFactorWith2SampleStates::Create(const ImuState &i1, const ImuState &i2, const ImuState &i3, double sp1_timestamp, double sp2_timestamp, double weight_gyr, double weight_acc, double weight_bg, double weight_ba, double dt, const Vector3d &gravity) {
  return (new ceres::AutoDiffCostFunction<ImuFactorWith2SampleStates, 6, 6, 6, 6>(
      new ImuFactorWith2SampleStates(i1, i2, i3, sp1_timestamp, sp2_timestamp, weight_gyr, weight_acc, weight_bg, weight_ba, dt, gravity)));
}

template <typename T>
void ImuFactorWith2SampleStates::ComputeStateCorr(const Eigen::Map<const Vector6<T>> &sp1, const Eigen::Map<const Vector6<T>> &sp2, double sp1_timestamp, double sp2_timestamp, double timestamp, Eigen::Quaternion<T> &R_cor, Eigen::Matrix<T, 3, 1> &t_cor) const {
  CHECK(timestamp >= sp1_timestamp && timestamp <= sp2_timestamp) << std::fixed << std::setprecision(9) << "Timestamp order: " << timestamp << " sp1: " << sp1_timestamp << " sp2: " << sp2_timestamp;

  double factor = (timestamp - sp1_timestamp) / (sp2_timestamp - sp1_timestamp);
  // interpolate rot
  Sophus::SO3<T> spl_rot = Sophus::SO3<T>::exp(sp1.template head<3>());
  Sophus::SO3<T> spr_rot = Sophus::SO3<T>::exp(sp2.template head<3>());
  R_cor                  = (spl_rot * Sophus::SO3<T>::exp(factor * (spl_rot.inverse() * spr_rot).log())).unit_quaternion();
  // interpolate pos and bias
  Vector6<T> sp = (1 - factor) * sp1 + factor * sp2;
  t_cor         = sp.template block<3, 1>(3, 0);
}

SurfelMatchBinaryFactor4SampleStates::SurfelMatchBinaryFactor4SampleStates(std::shared_ptr<Surfel> s1, std::shared_ptr<SampleState> sp1l, std::shared_ptr<SampleState> sp1r, std::shared_ptr<Surfel> s2, std::shared_ptr<SampleState> sp2l, std::shared_ptr<SampleState> sp2r) : s1_(s1), sp1l_(sp1l), sp1r_(sp1r), s2_(s2), sp2l_(sp2l), sp2r_(sp2r) {
  Matrix3d                                cov = s1_->covariance + s2_->covariance;
  Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
  weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
  norm_   = es.eigenvectors().col(0);
}

bool SurfelMatchBinaryFactor4SampleStates::Evaluate(double const *const *parameters,
                                                    double              *residuals,
                                                    double             **jacobians) const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp1l_cor{parameters[0]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp1r_cor{parameters[1]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp2l_cor{parameters[2]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp2r_cor{parameters[3]};

  double factor1 = (s1_->timestamp - sp1l_->timestamp) / (sp1r_->timestamp - sp1l_->timestamp);
  DCHECK_GE(factor1, 0.0);
  DCHECK_LE(factor1, 1.0);
  double factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
  DCHECK_GE(factor2, 0.0);
  DCHECK_LE(factor2, 1.0);

  residuals[0] = sym::SurfelBinaryMatch4SamplesWithJacobians0123(
      sp1l_cor, sp1r_cor, sp2l_cor, sp2r_cor,
      factor1, factor2, weight_, norm_, s1_->center, s2_->center,
      eps,
      jacobians ? jacobians[0] : nullptr,
      jacobians ? jacobians[1] : nullptr,
      jacobians ? jacobians[2] : nullptr,
      jacobians ? jacobians[3] : nullptr);

  return true;
}

SurfelMatchBinaryFactor3SampleStates::SurfelMatchBinaryFactor3SampleStates(std::shared_ptr<Surfel> s1, std::shared_ptr<SampleState> sp1l, std::shared_ptr<SampleState> sp1r, std::shared_ptr<Surfel> s2, std::shared_ptr<SampleState> sp2r) : s1_(s1), sp1l_(sp1l), sp1r_(sp1r), s2_(s2), sp2l_(sp1r), sp2r_(sp2r) {
  Matrix3d                                cov = s1_->covariance + s2_->covariance;
  Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
  weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
  norm_   = es.eigenvectors().col(0);
}

bool SurfelMatchBinaryFactor3SampleStates::Evaluate(double const *const *parameters,
                                                    double              *residuals,
                                                    double             **jacobians) const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp1l_cor{parameters[0]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp1r_cor{parameters[1]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp2r_cor{parameters[2]};

  double factor1 = (s1_->timestamp - sp1l_->timestamp) / (sp1r_->timestamp - sp1l_->timestamp);
  DCHECK_GE(factor1, 0.0);
  DCHECK_LE(factor1, 1.0);
  double factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
  DCHECK_GE(factor2, 0.0);
  DCHECK_LE(factor2, 1.0);

  residuals[0] = sym::SurfelBinaryMatch3SamplesWithJacobians012(
      sp1l_cor, sp1r_cor, sp2r_cor,
      factor1, factor2, weight_, norm_, s1_->center, s2_->center,
      eps,
      jacobians ? jacobians[0] : nullptr,
      jacobians ? jacobians[1] : nullptr,
      jacobians ? jacobians[2] : nullptr);

  return true;
}

SurfelMatchBinaryFactor2SampleStates::SurfelMatchBinaryFactor2SampleStates(std::shared_ptr<Surfel> s1, std::shared_ptr<SampleState> sp1l, std::shared_ptr<SampleState> sp1r, std::shared_ptr<Surfel> s2) : s1_(s1), sp1l_(sp1l), sp1r_(sp1r), s2_(s2), sp2l_(sp1l), sp2r_(sp1r) {
  Matrix3d                                cov = s1_->covariance + s2_->covariance;
  Eigen::SelfAdjointEigenSolver<Matrix3d> es(cov);
  weight_ = 1 / sqrt(pow(0.05 / 6, 2) + es.eigenvalues()[0]);
  norm_   = es.eigenvectors().col(0);
}

bool SurfelMatchBinaryFactor2SampleStates::Evaluate(double const *const *parameters,
                                                    double              *residuals,
                                                    double             **jacobians) const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp1l_cor{parameters[0]};
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> sp1r_cor{parameters[1]};

  double factor1 = (s1_->timestamp - sp1l_->timestamp) / (sp1r_->timestamp - sp1l_->timestamp);
  DCHECK_GE(factor1, 0.0);
  DCHECK_LE(factor1, 1.0);
  double factor2 = (s2_->timestamp - sp2l_->timestamp) / (sp2r_->timestamp - sp2l_->timestamp);
  DCHECK_GE(factor2, 0.0);
  DCHECK_LE(factor2, 1.0);

  residuals[0] = sym::SurfelBinaryMatch2SamplesWithJacobians01(
      sp1l_cor, sp1r_cor,
      factor1, factor2, weight_, norm_, s1_->center, s2_->center,
      eps,
      jacobians ? jacobians[0] : nullptr,
      jacobians ? jacobians[1] : nullptr);

  return true;
}
