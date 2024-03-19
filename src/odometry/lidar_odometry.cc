
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(6, 0, " ", "\n", "", "")
#define _GLIBCXX_ASSERTIONS

#include <absl/container/flat_hash_map.h>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>

#include "common/histogram.h"
#include "common/utils.h"
#include "knn_surfel_matcher.h"
#include "odometry/cost_functor.h"
#include "odometry/lidar_odometry.h"
#include "odometry/local_parameterization_sample_state.h"
#include "odometry/spline_interpolation.h"
#include "surfel_extraction.h"

namespace {

class CubicBSplineSampleCorrector {
 public:
  CubicBSplineSampleCorrector(const std::deque<SampleState::Ptr> &sample_states) {
    std::vector<double>   sample_timestamps;
    std::vector<Vector3d> sample_rot;
    std::vector<Vector3d> sample_pos;
    for (auto &sample_state : sample_states) {
      sample_timestamps.push_back(sample_state->timestamp);
      sample_rot.emplace_back(sample_state->rot_cor);
      sample_pos.emplace_back(sample_state->pos_cor);
    }
    rot_interp_.reset(new CubicBSplineInterpolator(sample_timestamps, sample_rot));
    pos_interp_.reset(new CubicBSplineInterpolator(sample_timestamps, sample_pos));
  }

  bool GetCorr(double timestamp, Vector3d &rot_cor, Vector3d &pos_cor) {
    CHECK(rot_interp_ && pos_interp_) << "Interpolator not initialized";
    auto rot_cor_ptr = rot_interp_->Interp(timestamp);
    auto pos_cor_ptr = pos_interp_->Interp(timestamp);
    CHECK((rot_cor_ptr && pos_cor_ptr) || (!rot_cor_ptr && !pos_cor_ptr)) << "Interpolation failed";
    if (rot_cor_ptr) {
      rot_cor = *rot_cor_ptr;
      pos_cor = *pos_cor_ptr;
      return true;
    } else {
      return false;
    }
  }

 private:
  std::shared_ptr<CubicBSplineInterpolator> rot_interp_;
  std::shared_ptr<CubicBSplineInterpolator> pos_interp_;
};

void PrintSurfelResiduals(const std::vector<ceres::ResidualBlockId> &residual_ids, ceres::Problem &problem, const std::string &window_type) {
  if (residual_ids.empty()) {
    return;
  }
  std::vector<double>             residuals;
  double                          cost;
  ceres::Problem::EvaluateOptions options;
  options.apply_loss_function = true;
  options.residual_blocks     = residual_ids;
  problem.Evaluate(options, &cost, &residuals, nullptr, nullptr);
  Histogram hist;
  for (auto &e : residuals) {
    hist.Add(e);
  }
  LOG(INFO) << window_type << " Surfel residuals, cost: " << cost << ", dist: " << hist.ToString(10);
}

void PrintImuResiduals(const std::vector<ceres::ResidualBlockId> &residual_ids, ceres::Problem &problem) {
  if (residual_ids.empty()) {
    return;
  }
  std::vector<double>             residuals;
  double                          cost;
  ceres::Problem::EvaluateOptions options;
  options.apply_loss_function = true;
  options.residual_blocks     = residual_ids;
  problem.Evaluate(options, &cost, &residuals, nullptr, nullptr);
  Histogram                hist[4];
  std::vector<std::string> residual_types = {"gyro", "acc", "gyro_bias", "acc_bias"};
  for (int i = 0; i < residuals.size(); i += 12) {
    for (int j = 0; j < 4; ++j) {
      auto residuals_part = Vector3d{residuals[i + j * 3], residuals[i + j * 3 + 1], residuals[i + j * 3 + 2]};
      hist[j].Add(residuals_part.norm());
    }
  }
  for (int j = 0; j < 4; ++j) {
    LOG(INFO) << "Imu residuals with type " << residual_types[j] << ", cost: " << cost << ", dist: " << hist[j].ToString(10);
  }
}

void PrintSampleStates(const std::deque<SampleState::Ptr> &states) {
  for (auto &e : states) {
    LOG(INFO) << "\np:  " << e->pos.transpose() << "\nDp: " << e->pos_cor.transpose() << "\nq:  " << e->rot.coeffs().transpose() << "\nbg: " << e->bg.transpose() << "\nba: " << e->ba.transpose();
  }
}

/**
 * @brief Predict pose of a new imu state
 *
 * @param i1
 * @param i2
 * @param ba
 * @param bg
 * @param grav
 * @param i3 timestamp, acc and gyr has been set before calling this function
 */
void PredictPoseOfNewImuState(
    const ImuState &i1,
    const ImuState &i2,
    const Vector3d &ba,
    const Vector3d &bg,
    const Vector3d &grav,
    ImuState       &i3) {
  CHECK_NEAR(i3.timestamp - i2.timestamp, i2.timestamp - i1.timestamp, 1e-6);
  double dt = i3.timestamp - i2.timestamp;
  i3.rot    = i2.rot * Exp(((i2.gyr + i3.gyr) / 2 - bg) * dt);
  i3.pos    = (i1.rot * (i1.acc - ba) + grav) * dt * dt + 2 * i2.pos - i1.pos;
}

/**
 * @brief Build a sweep from points_buff
 *
 * Timestamp order: l_0 < l_1 < ... < l_{n-1} < lidar_end_time
 *
 * @param points_buff
 * @param sweep_endtime
 * @param sweep
 */
void BuildSweep(std::deque<hilti_ros::Point> &points_buff, double sweep_endtime, std::vector<hilti_ros::Point> &sweep) {
  sweep.clear();
  while (!points_buff.empty() && points_buff.front().time < sweep_endtime) {
    sweep.push_back(points_buff.front());
    points_buff.pop_front();
  }
}

void UndistortSweep(const std::vector<hilti_ros::Point> &sweep_in,
                    const std::deque<ImuState>          &imu_states,
                    std::vector<hilti_ros::Point>       &sweep_out) {
  sweep_out.clear();
  for (auto &pt : sweep_in) {
    auto it  = std::lower_bound(imu_states.begin(), imu_states.end(), pt.time, [](const ImuState &a, auto b) { return a.timestamp < b; });
    auto idx = it - imu_states.begin();
    CHECK(idx >= 1 && idx < imu_states.size()) << idx;
    double      factor      = (pt.time - imu_states[idx - 1].timestamp) / (imu_states[idx].timestamp - imu_states[idx - 1].timestamp);
    Vector3d    pos         = imu_states[idx - 1].pos * (1 - factor) + imu_states[idx].pos * factor;
    Quaterniond rot         = imu_states[idx - 1].rot.slerp(factor, imu_states[idx].rot);
    auto        new_pt      = pt;
    new_pt.getVector3fMap() = (rot * new_pt.getVector3fMap().cast<double>() + pos).cast<float>();
    sweep_out.push_back(new_pt);
  }
}

void UpdateSurfelPoses(const std::deque<ImuState> &imu_states, std::deque<Surfel::Ptr> &surfels) {
  for (auto &surfel : surfels) {
    auto it  = std::lower_bound(imu_states.begin(), imu_states.end(), surfel->timestamp, [](const ImuState &a, auto b) { return a.timestamp < b; });
    auto idx = it - imu_states.begin();
    CHECK(idx != 0 && idx != imu_states.size()) << idx;
    double      factor = (surfel->timestamp - imu_states[idx - 1].timestamp) / (imu_states[idx].timestamp - imu_states[idx - 1].timestamp);
    Vector3d    pos    = imu_states[idx - 1].pos * (1 - factor) + imu_states[idx].pos * factor;
    Quaterniond rot    = imu_states[idx - 1].rot.slerp(factor, imu_states[idx].rot);
    surfel->UpdatePose(pos, rot);
  }
}

void UpdateSamplePoses(std::deque<SampleState::Ptr> &sample_states) {
  for (auto &sample_state : sample_states) {
    sample_state->rot = Exp(sample_state->rot_cor) * sample_state->rot;
    sample_state->pos = sample_state->pos_cor + sample_state->pos;
    sample_state->rot_cor.setZero();
    sample_state->pos_cor.setZero();
  }
}

/**
 * @brief Update imu poses by sample state corrections
 *
 * @param sample_states
 * @param imu_states
 */
void UpdateImuPoses(const std::deque<SampleState::Ptr> &sample_states,
                    std::deque<ImuState>               &imu_states) {
  int                         corrected_first_idx = -1, corrected_last_idx = -1;
  CubicBSplineSampleCorrector corrector(sample_states);
  // update imu poses by sample state corrections
  for (int i = 0; i < imu_states.size(); ++i) {
    auto    &imu_state = imu_states[i];
    Vector3d rot_cor, pos_cor;
    bool     interp_ok = corrector.GetCorr(imu_state.timestamp, rot_cor, pos_cor);
    if (interp_ok) {
      imu_state.rot = Exp(rot_cor) * imu_state.rot;
      imu_state.pos = pos_cor + imu_state.pos;

      if (corrected_first_idx == -1) corrected_first_idx = i;
      corrected_last_idx = i;
    }
  }
  // update heading and tailing imu poses
  if (corrected_first_idx != -1) {
    LOG(INFO) << "Correct extra imu poses in ("
              << corrected_last_idx << ","
              << imu_states.size() << ")";
    CHECK_EQ(corrected_first_idx, 0);
    CHECK_EQ(corrected_last_idx, imu_states.size() - 2);

    int size = imu_states.size();
    PredictPoseOfNewImuState(imu_states[size - 3], imu_states[size - 2], sample_states.back()->ba, sample_states.back()->bg, sample_states.back()->grav, imu_states[size - 1]);
  }
}

/**
 * @brief Trim to sliding window
 *
 * Timestamp order: sample_0 <= imu_0 <= surfel_0
 *
 * @param sample_states
 * @param imu_states
 * @param surfels_sld_win
 * @param surfels_fix_win
 * @param window_duration
 * @param io
 */
void ShrinkToFit(std::deque<SampleState::Ptr> &sample_states,
                 std::deque<ImuState>         &imu_states,
                 std::deque<Surfel::Ptr>      &surfels_sld_win,
                 std::deque<Surfel::Ptr>      &surfels_fix_win,
                 double                        sld_win_duration,
                 double                        fix_win_duration,
                 OdometryIO                   &io) {
  if (sample_states.empty() || sample_states.back()->timestamp - sample_states.front()->timestamp <= sld_win_duration) {
    return;
  }
  while (sample_states.back()->timestamp - sample_states.front()->timestamp > sld_win_duration) {
    sample_states.pop_front();
  }
  while (imu_states.front().timestamp < sample_states.front()->timestamp) {
    auto &imu_state = imu_states.front();
    io.AddOdom(imu_state.timestamp, imu_state.rot, imu_state.pos);
    imu_states.pop_front();
  }
  while (surfels_sld_win.front()->timestamp < imu_states.front().timestamp) {
    surfels_fix_win.push_front(surfels_sld_win.front());
    surfels_sld_win.pop_front();
  }
  while (surfels_fix_win.back()->timestamp - surfels_fix_win.back()->timestamp > fix_win_duration) {
    surfels_fix_win.pop_back();
  }
}

}  // namespace

void LidarOdometry::BuildSldWinLidarResiduals(const std::vector<SurfelCorrespondence> &surfel_corrs, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids) {
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> correlation_matrix;
  correlation_matrix.resize(sample_states_sld_win_.size() - 1, sample_states_sld_win_.size() - 1);
  correlation_matrix.setZero();

  for (const auto &surfel_corr : surfel_corrs) {
    CHECK_LT(surfel_corr.s1->timestamp, surfel_corr.s2->timestamp) << std::fixed << std::setprecision(6) << surfel_corr.s1->timestamp << " " << surfel_corr.s2->timestamp;  // bug: disorder happens

    auto sp1r_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), surfel_corr.s1->timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    CHECK(sp1r_it != sample_states_sld_win_.begin());
    CHECK(sp1r_it != sample_states_sld_win_.end());

    auto sp1l    = *(sp1r_it - 1);
    auto sp1r    = *(sp1r_it);
    auto sp2r_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), surfel_corr.s2->timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    CHECK(sp2r_it != sample_states_sld_win_.begin());
    CHECK(sp2r_it != sample_states_sld_win_.end());
    auto sp2l = *(sp2r_it - 1);
    auto sp2r = *(sp2r_it);

    auto loss_function = new ceres::CauchyLoss(0.4);  // todo set cauchy loss param
    if (sp1r->timestamp < sp2l->timestamp) {
      auto residual_id = problem.AddResidualBlock(
          SurfelMatchBinaryFactor::Create(surfel_corr.s1, sp1l, sp1r, surfel_corr.s2, sp2l, sp2r),
          loss_function,
          sp1l->data_cor,
          sp1r->data_cor,
          sp2l->data_cor,
          sp2r->data_cor);
      residual_ids.push_back(residual_id);
    } else if (sp1r == sp2l) {
      auto residual_id = problem.AddResidualBlock(
          SurfelMatchBinaryFactor::Create(surfel_corr.s1, sp1l, sp1r, surfel_corr.s2, sp2r),
          loss_function,
          sp1l->data_cor,
          sp1r->data_cor,
          sp2r->data_cor);
      residual_ids.push_back(residual_id);
    } else {
      auto residual_id = problem.AddResidualBlock(
          SurfelMatchBinaryFactor::Create(surfel_corr.s1, sp1l, sp1r, surfel_corr.s2),
          loss_function,
          sp1l->data_cor,
          sp1r->data_cor);
      residual_ids.push_back(residual_id);
    }

    int sp1l_idx = std::distance(sample_states_sld_win_.begin(), sp1r_it) - 1;
    int sp2l_idx = std::distance(sample_states_sld_win_.begin(), sp2r_it) - 1;
    correlation_matrix(sp1l_idx, sp2l_idx) += 1;
    correlation_matrix(sp2l_idx, sp1l_idx) += 1;
  }
  // for (int i = 0; i < correlation_matrix.rows(); ++i) {
  //   for (int j = 0; j < correlation_matrix.cols(); ++j) {
  //     std::cout << correlation_matrix(i, j) << " ";
  //   }
  //   std::cout << std::endl;
  // }
}

void LidarOdometry::BuildFixWinLidarResiduals(const std::vector<SurfelCorrespondence> &surfel_corrs, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids) {
  for (const auto &surfel_corr : surfel_corrs) {
    CHECK_LT(surfel_corr.s1->timestamp, surfel_corr.s2->timestamp) << std::fixed << std::setprecision(6) << surfel_corr.s1->timestamp << " " << surfel_corr.s2->timestamp;  // bug: disorder happens

    auto sp2r_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), surfel_corr.s2->timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    CHECK(sp2r_it != sample_states_sld_win_.begin());
    CHECK(sp2r_it != sample_states_sld_win_.end());
    auto sp2l = *(sp2r_it - 1);
    auto sp2r = *(sp2r_it);

    auto loss_function = new ceres::CauchyLoss(0.4);  // todo set cauchy loss param
    auto residual_id   = problem.AddResidualBlock(
        SurfelMatchUnaryFactor::Create(surfel_corr.s1, surfel_corr.s2, sp2l, sp2r),
        loss_function,
        sp2l->data_cor,
        sp2r->data_cor);
    residual_ids.push_back(residual_id);
  }
}

void LidarOdometry::BuildImuResiduals(const std::deque<ImuState> &imu_states, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids) {
  for (int i = 0; i < imu_states.size() - 2; ++i) {
    auto &i1 = imu_states[i];
    auto &i2 = imu_states[i + 1];
    auto &i3 = imu_states[i + 2];
    if (i1.timestamp < sample_states_sld_win_.front()->timestamp) {
      continue;
    }
    if (i3.timestamp > sample_states_sld_win_.back()->timestamp) {
      break;
    }
    auto sp2_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), i1.timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    auto sp1    = *(sp2_it - 1);
    auto sp2    = *(sp2_it);
    if (sp2_it == sample_states_sld_win_.end() - 1) {
      auto residual_id = problem.AddResidualBlock(
          ImuFactorWith2SampleStates::Create(
              i1, i2, i3,
              sp1->timestamp, sp2->timestamp,
              config_.gyroscope_noise_density_cost_weight,
              config_.accelerometer_noise_density_cost_weight,
              config_.gyroscope_random_walk_cost_weight,
              config_.accelerometer_random_walk_cost_weight,
              1 / config_.imu_rate, sample_states_sld_win_.back()->grav),
          new ceres::TrivialLoss(),  // todo use loss function
          sp1->data_cor,
          sp2->data_cor);
      residual_ids.push_back(residual_id);
    } else {
      auto sp3         = *(sp2_it + 1);
      auto residual_id = problem.AddResidualBlock(
          ImuFactorWith3SampleStates::Create(
              i1, i2, i3,
              sp1->timestamp, sp2->timestamp, sp3->timestamp,
              config_.gyroscope_noise_density_cost_weight,
              config_.accelerometer_noise_density_cost_weight,
              config_.gyroscope_random_walk_cost_weight,
              config_.accelerometer_random_walk_cost_weight,
              1 / config_.imu_rate, sample_states_sld_win_.back()->grav),
          new ceres::TrivialLoss(),
          sp1->data_cor,
          sp2->data_cor,
          sp3->data_cor);
      residual_ids.push_back(residual_id);
    }
  }
}

void LidarOdometry::PredictImuStatesAndSampleStates(double end_time) {
  // 1. try to initialize imu states and sample states
  CHECK_GE(imu_buff_.size(), 2);
  auto        dt           = 1 / config_.imu_rate;
  static bool init_sld_win = false;
  if (!init_sld_win) {
    for (int i = 0; i < 2; ++i) {
      auto imu_msg = imu_buff_.front();
      imu_buff_.pop_front();

      ImuState imu_state;
      imu_state.timestamp = imu_msg.timestamp;
      imu_state.acc       = imu_msg.linear_acceleration;
      imu_state.gyr       = imu_msg.angular_velocity;
      imu_state.pos       = Vector3d::Zero();
      if (i == 0) {
        imu_state.rot = Quaterniond::Identity();
      } else {
        imu_state.rot = Exp((imu_states_sld_win_.back().gyr + imu_state.gyr) / 2 * dt);
      }
      imu_states_sld_win_.push_back(imu_state);
    }

    SampleState::Ptr ss(new SampleState);
    ss->timestamp = imu_states_sld_win_.front().timestamp;
    ss->ba.setZero();
    ss->bg.setZero();
    ss->grav = -config_.gravity_norm * imu_states_sld_win_.front().acc.normalized();
    ss->rot  = imu_states_sld_win_.front().rot;
    ss->pos  = imu_states_sld_win_.front().pos;
    sample_states_sld_win_.push_back(ss);

    init_sld_win = true;
  }

  auto   sample_states_old_size     = sample_states_sld_win_.size();
  double sample_states_old_lasttime = sample_states_sld_win_.back()->timestamp;
  int    sample_states_add_size     = (end_time - sample_states_old_lasttime) / config_.sample_dt;
  double sample_states_add_lasttime = sample_states_old_lasttime + config_.sample_dt * sample_states_add_size;

  // 2. predict imu states
  auto ba   = sample_states_sld_win_.back()->ba;
  auto bg   = sample_states_sld_win_.back()->bg;
  auto grav = sample_states_sld_win_.back()->grav;
  while (!imu_buff_.empty()) {
    int size = imu_states_sld_win_.size();

    auto imu_msg = imu_buff_.front();
    imu_buff_.pop_front();

    ImuState imu_state;
    imu_state.timestamp = imu_msg.timestamp;
    imu_state.acc       = imu_msg.linear_acceleration;
    imu_state.gyr       = imu_msg.angular_velocity;
    PredictPoseOfNewImuState(imu_states_sld_win_[size - 2], imu_states_sld_win_[size - 1], ba, bg, grav, imu_state);

    imu_states_sld_win_.push_back(imu_state);

    if (imu_state.timestamp >= sample_states_add_lasttime) {
      // ensure that we have enough imu states
      break;
    }
  }

  // 3. add more sample states
  for (int i = 1; i <= sample_states_add_size; ++i) {
    double timestamp = sample_states_old_lasttime + i * config_.sample_dt;

    SampleState::Ptr ss(new SampleState);
    ss->timestamp = timestamp;
    ss->ba        = ba;
    ss->bg        = bg;
    ss->grav      = grav;

    auto it  = std::lower_bound(imu_states_sld_win_.begin(), imu_states_sld_win_.end(), timestamp, [&](const ImuState &a, double b) {
      return a.timestamp < b;
    });
    auto idx = it - imu_states_sld_win_.begin();

    CHECK_NE(idx, 0);
    CHECK_NE(idx, imu_states_sld_win_.size());

    double factor = (timestamp - imu_states_sld_win_[idx - 1].timestamp) / (imu_states_sld_win_[idx].timestamp - imu_states_sld_win_[idx - 1].timestamp);
    ss->rot       = imu_states_sld_win_[idx - 1].rot.slerp(factor, imu_states_sld_win_[idx].rot);
    ss->pos       = (1 - factor) * imu_states_sld_win_[idx - 1].pos + factor * imu_states_sld_win_[idx].pos;
    CHECK_GE(factor, 0);
    CHECK_LE(factor, 1);
    sample_states_sld_win_.push_back(ss);
  }
  LOG(INFO) << std::fixed << std::setprecision(6) << "Adding sample states_" << sample_states_sld_win_.size() - sample_states_old_size << "(" << sample_states_old_lasttime << "," << sample_states_sld_win_.back()->timestamp << "]";
}

bool LidarOdometry::SyncHeadingMsgs() {
  static bool sync_done = false;
  if (sync_done) {
    return true;
  }

  if (imu_buff_.empty() || points_buff_.empty()) {
    return false;
  }

  if (imu_buff_.back().timestamp < points_buff_.front().time) {
    LOG(INFO) << "waiting for imu message...";
    return false;
  }

  while (imu_buff_.front().timestamp < points_buff_.front().time) {
    imu_buff_.pop_front();
    CHECK(!imu_buff_.empty());
  }

  while (points_buff_.front().time < imu_buff_.front().timestamp) {
    points_buff_.pop_front();
    CHECK(!points_buff_.empty());
  }

  sync_done = true;

  return true;
}

void LidarOdometry::AddLidarScan(const pcl::PointCloud<hilti_ros::Point>::Ptr &msg) {
  // transform points from lidar frame to imu frame
  for (auto pt : *msg) {
    pt.getVector3fMap() = (this->config_.ext_lidar2imu * pt.getVector3fMap().cast<double>()).cast<float>();
    CHECK(points_buff_.empty() || pt.time >= points_buff_.back().time);
    if (pt.getVector3fMap().norm() < config_.min_range || pt.getVector3fMap().norm() > config_.max_range || config_.blind_bounding_box.contains(pt.getVector3fMap().cast<double>())) {
      continue;
    }
    points_buff_.push_back(pt);
  }

  if (!SyncHeadingMsgs()) {
    return;
  }

  // 1. collect scan to sweep
  std::vector<hilti_ros::Point> sweep;
  auto                          sweep_endtime = points_buff_.front().time + config_.sweep_duration;
  if (points_buff_.back().time < sweep_endtime || imu_buff_.empty() ||
      imu_buff_.back().timestamp < sweep_endtime) {
    // LOG(INFO) << "Waiting to construct a sweep: " << points_buff_.back().time - points_buff_.front().time;
    return;
  }

  // 2. integrate IMU poses in windows
  PredictImuStatesAndSampleStates(sweep_endtime);
  sweep_endtime = sample_states_sld_win_.back()->timestamp;  // todo here we can make sure all points/surfels are before sweep_endtime

  BuildSweep(points_buff_, sweep_endtime, sweep);
  LOG(INFO) << std::fixed << std::setprecision(6) << "Build sweep " << sweep_id_ << " with points_" << sweep.size() << "[" << sweep.front().time << "," << sweep.back().time << "] by sweep_endtime " << sweep_endtime;

  // 3. undistort sweep by IMU poses
  std::vector<hilti_ros::Point> sweep_undistorted;
  UndistortSweep(sweep, imu_states_sld_win_, sweep_undistorted);

  // 4. extract surfels and add to windows, the first time surfels will be add to global map
  std::deque<Surfel::Ptr> surfels_sweep;
  GlobalMap               map;
  BuildSurfels(sweep_undistorted, surfels_sweep, map);
  surfels_sld_win_.insert(surfels_sld_win_.end(), surfels_sweep.begin(), surfels_sweep.end());
  UpdateSurfelPoses(imu_states_sld_win_, surfels_sld_win_);

  for (int iter_num = 0; iter_num < config_.outer_iter_num_max; ++iter_num) {
    std::vector<SurfelCorrespondence> surfel_corrs_sld, surfel_corrs_fix;

    KnnSurfelMatcher surfel_matcher_sld_win;
    surfel_matcher_sld_win.BuildIndex(surfels_sld_win_);
    surfel_matcher_sld_win.Match(surfels_sld_win_, surfel_corrs_sld);

    KnnSurfelMatcher surfel_matcher_fix_win;
    surfel_matcher_fix_win.BuildIndex(surfels_fix_win_);
    surfel_matcher_fix_win.Match(surfels_sld_win_, surfel_corrs_fix);

    // 5. sovle poses in windows
    ceres::Problem                      problem;
    std::vector<ceres::ResidualBlockId> surfel_sld_win_residual_ids, surfel_fix_win_residual_ids, imu_residual_ids;
    BuildSldWinLidarResiduals(surfel_corrs_sld, problem, surfel_sld_win_residual_ids);
    BuildFixWinLidarResiduals(surfel_corrs_fix, problem, surfel_fix_win_residual_ids);
    BuildImuResiduals(imu_states_sld_win_, problem, imu_residual_ids);

    PrintSurfelResiduals(surfel_sld_win_residual_ids, problem, "Sliding Window");
    PrintSurfelResiduals(surfel_fix_win_residual_ids, problem, "Fixed Window");
    PrintImuResiduals(imu_residual_ids, problem);

    ceres::Solver::Options option;
    option.minimizer_progress_to_stdout = true;
    option.linear_solver_type           = ceres::DENSE_SCHUR;
    option.max_num_iterations           = config_.inner_iter_num_max;
    option.num_threads                  = 8;
    ceres::Solver::Summary summary;
    static auto            g_first_sample_state = sample_states_sld_win_[0];
    for (auto &sample_state : sample_states_sld_win_) {
      if (sample_state == g_first_sample_state) {
        LOG(INFO) << "Optimize with fixing position of the first sample state.";
        problem.SetParameterization(sample_state->data_cor, LocalParameterizationSampleStateFixPos::Create());
      } else {
        problem.SetParameterization(sample_state->data_cor, LocalParameterizationSampleState::Create());
      }
    }
    ceres::Solve(option, &problem, &summary);
    LOG(INFO) << summary.BriefReport();

    UpdateImuPoses(sample_states_sld_win_, imu_states_sld_win_);
    UpdateSurfelPoses(imu_states_sld_win_, surfels_sld_win_);
    UpdateSamplePoses(sample_states_sld_win_);

    PrintSurfelResiduals(surfel_sld_win_residual_ids, problem, "Sliding Window");
    PrintSurfelResiduals(surfel_fix_win_residual_ids, problem, "Fixed Window");
    PrintImuResiduals(imu_residual_ids, problem);
    PrintSampleStates(sample_states_sld_win_);
  }

  ShrinkToFit(
      sample_states_sld_win_,
      imu_states_sld_win_,
      surfels_sld_win_,
      surfels_fix_win_,
      config_.sliding_window_duration,
      config_.fixed_window_duration,
      io_);

  PubSurfels(surfels_sld_win_, pub_plane_map_);
  {
    std::vector<hilti_ros::Point> sweep_undistorted_final;
    UndistortSweep(sweep, imu_states_sld_win_, sweep_undistorted_final);
    sensor_msgs::PointCloud2          msg;
    pcl::PointCloud<hilti_ros::Point> cloud;
    for (auto &e : sweep_undistorted_final) {
      cloud.push_back(e);
    }
    pcl::toROSMsg(cloud, msg);
    msg.header.stamp.fromSec(cloud.points[0].time);
    msg.header.frame_id = "world";
    pub_scan_in_imu_frame_.publish(msg);
  }
  {
    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    transform.setOrigin(tf::Vector3(sample_states_sld_win_.back()->pos[0], sample_states_sld_win_.back()->pos[1], sample_states_sld_win_.back()->pos[2]));
    transform.setRotation(tf::Quaternion(sample_states_sld_win_.back()->rot.x(), sample_states_sld_win_.back()->rot.y(), sample_states_sld_win_.back()->rot.z(), sample_states_sld_win_.back()->rot.w()));
    br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(sample_states_sld_win_.back()->timestamp), "world", "imu_link"));
  }

  ++sweep_id_;
}

void LidarOdometry::AddImuData(const ImuData &msg) {
  auto msg_new = msg;
  // msg_new.angular_velocity += Vector3d::Constant(0.02);
  this->imu_buff_.push_back(msg_new);
}

LidarOdometry::LidarOdometry() : io_("/tmp") {
  pub_plane_map_         = nh_.advertise<visualization_msgs::MarkerArray>("/current_planes", 10);
  pub_scan_in_imu_frame_ = nh_.advertise<sensor_msgs::PointCloud2>("/scan_in_imu_frame", 10);
}

LidarOdometry::~LidarOdometry() {
  for (auto &e : imu_states_sld_win_) {
    io_.AddOdom(e.timestamp, e.rot, e.pos);
  }
}
