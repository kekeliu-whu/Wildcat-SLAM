
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(9, 0, " ", "\n", "", "")
#define _GLIBCXX_ASSERTIONS

#include <absl/container/flat_hash_map.h>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <nav_msgs/Path.h>
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
  std::vector<std::string> residual_types = {"gyro", "acc"};
  for (int i = 0; i < residuals.size(); i += 6) {
    for (int j = 0; j < 2; ++j) {
      auto residuals_part = Vector3d{residuals[i + j * 3], residuals[i + j * 3 + 1], residuals[i + j * 3 + 2]};
      hist[j].Add(residuals_part.norm());
    }
  }
  for (int j = 0; j < 2; ++j) {
    LOG(INFO) << std::fixed << std::setprecision(9) << "Imu residuals with type " << residual_types[j] << ", cost: " << cost << ", dist: " << hist[j].ToString(10);
  }
}

void PrintSampleStates(const std::deque<SampleState::Ptr> &states) {
  for (auto &e : states) {
    DLOG(INFO) << "\npos_cor: " << e->pos_cor.transpose() << "\nrot_cor: " << e->rot_cor.transpose() << "\nbg: " << e->bg.transpose() << "\nba: " << e->ba.transpose();
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

void UndistortSweep(const std::deque<hilti_ros::Point> &sweep_in,
                    const std::deque<ImuState>         &imu_states,
                    std::deque<hilti_ros::Point>       &sweep_out) {
  std::deque<hilti_ros::Point> sweep_out_real;
  sweep_out_real.clear();
  for (auto &pt : sweep_in) {
    auto it  = std::lower_bound(imu_states.begin(), imu_states.end(), pt.time, [](const ImuState &a, auto b) { return a.timestamp < b; });
    auto idx = it - imu_states.begin();
    CHECK(idx >= 1 && idx < imu_states.size()) << idx;
    double      factor      = (pt.time - imu_states[idx - 1].timestamp) / (imu_states[idx].timestamp - imu_states[idx - 1].timestamp);
    Vector3d    pos         = imu_states[idx - 1].pos * (1 - factor) + imu_states[idx].pos * factor;
    Quaterniond rot         = imu_states[idx - 1].rot.slerp(factor, imu_states[idx].rot);
    auto        new_pt      = pt;
    new_pt.getVector3fMap() = (rot * new_pt.getVector3fMap().cast<double>() + pos).cast<float>();
    sweep_out_real.push_back(new_pt);
  }
  sweep_out = sweep_out_real;
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

/**
 * @brief Update surfel normal based on the view point
 *
 * @param imu_states
 * @param surfels
 */
void UpdateSurfelNormals(const std::deque<ImuState> &imu_states, std::deque<Surfel::Ptr> &surfels) {
  for (auto &surfel : surfels) {
    auto it  = std::lower_bound(imu_states.begin(), imu_states.end(), surfel->timestamp, [](const ImuState &a, auto b) { return a.timestamp < b; });
    auto idx = it - imu_states.begin();
    CHECK(idx != 0 && idx != imu_states.size()) << idx;
    double   factor     = (surfel->timestamp - imu_states[idx - 1].timestamp) / (imu_states[idx].timestamp - imu_states[idx - 1].timestamp);
    Vector3d view_point = imu_states[idx - 1].pos * (1 - factor) + imu_states[idx].pos * factor;
    if (surfel->normal.dot(surfel->center - view_point) < 0) {
      surfel->normal = -surfel->normal;
    }
  }
}

void UpdateSamplePoses(std::deque<SampleState::Ptr> &sample_states) {
  for (auto &sample_state : sample_states) {
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
    CHECK(interp_ok);
    imu_state.rot = Exp(rot_cor) * imu_state.rot;
    imu_state.pos = pos_cor + Exp(rot_cor) * imu_state.pos;
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
                 std::deque<ImuState>         &imu_states_sld_win,
                 std::deque<ImuState>         &imu_states_shift_out,
                 std::deque<hilti_ros::Point> &points_buff_sld_win,
                 std::deque<hilti_ros::Point> &points_buff_shift_out,
                 std::deque<Surfel::Ptr>      &surfels_sld_win,
                 std::deque<Surfel::Ptr>      &surfels_fix_win,
                 double                        sld_win_duration,
                 double                        fix_win_duration) {
  if (sample_states.empty() || sample_states.back()->timestamp - sample_states.front()->timestamp <= sld_win_duration) {
    return;
  }
  // shift out sample states
  while (sample_states.back()->timestamp - sample_states.front()->timestamp > sld_win_duration) {
    sample_states.pop_front();
  }
  // shift out imu states
  while (imu_states_sld_win.front().timestamp < sample_states.front()->timestamp) {
    imu_states_shift_out.push_back(imu_states_sld_win.front());
    imu_states_sld_win.pop_front();
  }
  // shift out sld win surfels into fix win
  while (true) {
    points_buff_shift_out.push_back(points_buff_sld_win.front());
    points_buff_sld_win.pop_front();
    if (points_buff_sld_win.front().time > imu_states_sld_win.front().timestamp) {
      break;
    }
  }
  // shift out sld win points
  while (surfels_sld_win.front()->timestamp < imu_states_sld_win.front().timestamp) {
    surfels_fix_win.push_back(surfels_sld_win.front());
    surfels_sld_win.pop_front();
  }
  // shift out fix win
  while (surfels_fix_win.back()->timestamp - surfels_fix_win.back()->timestamp > fix_win_duration) {
    surfels_fix_win.pop_back();
  }
}

/**
 * @brief Predict imu states and sample states
 *
 * Timestamp Order:
 *   IMU:          i_0 < i_1 < ... < i_{n-2} < end_time <= i_{n-1}
 *   Sample State: s_0 < s_1 < ... < s_{n-1} < end_time
 *
 * @param end_time
 */
void PredictImuStatesAndSampleStates(const MeasureGroup &mg, const LioConfig &config, std::deque<ImuState> &imu_states_sld_win, std::deque<SampleState::Ptr> &sample_states_sld_win) {
  auto imu_msgs = mg.imu_msgs;
  // 1. try to initialize imu states and sample states
  CHECK_GE(imu_msgs.size(), 2);
  auto        dt           = 1 / config.imu_rate;
  static bool init_sld_win = false;
  if (!init_sld_win) {
    for (int i = 0; i < 2; ++i) {
      auto imu_msg = imu_msgs.front();
      imu_msgs.pop_front();

      ImuState imu_state;
      imu_state.timestamp = imu_msg.timestamp;
      imu_state.acc       = imu_msg.linear_acceleration;
      imu_state.gyr       = imu_msg.angular_velocity;
      imu_state.pos       = Vector3d::Zero();
      if (i == 0) {
        imu_state.rot = Quaterniond::Identity();
      } else {
        imu_state.rot = Exp((imu_states_sld_win.back().gyr + imu_state.gyr) / 2 * dt);
      }
      imu_states_sld_win.push_back(imu_state);
    }

    SampleState::Ptr ss(new SampleState);
    ss->timestamp = imu_states_sld_win.front().timestamp;
    ss->ba.setZero();
    ss->bg.setZero();
    ss->grav = -config.gravity_norm * imu_states_sld_win.front().acc.normalized();
    sample_states_sld_win.push_back(ss);

    init_sld_win = true;
  }

  auto   sample_states_old_size     = sample_states_sld_win.size();
  double sample_states_old_lasttime = sample_states_sld_win.back()->timestamp;
  auto   ba                         = sample_states_sld_win.back()->ba;
  auto   bg                         = sample_states_sld_win.back()->bg;
  auto   grav                       = sample_states_sld_win.back()->grav;

  // 3. add more sample states
  for (int i = 1; i <= config.sample_num_per_sweep; ++i) {
    double timestamp = mg.imu_msgs.at(config.imu_num_per_sweep / config.sample_num_per_sweep * i).timestamp;

    SampleState::Ptr ss(new SampleState);
    ss->timestamp = timestamp;
    ss->grav      = grav;
    sample_states_sld_win.push_back(ss);
  }
  DLOG(INFO) << std::fixed << std::setprecision(9) << "Adding sample states_" << sample_states_sld_win.size() - sample_states_old_size << "(" << sample_states_old_lasttime << "," << sample_states_sld_win.back()->timestamp << "]";
  CHECK_EQ(imu_msgs.back().timestamp, sample_states_sld_win.back()->timestamp) << std::fixed << std::setprecision(9) << imu_msgs.back().timestamp << "," << sample_states_sld_win.back()->timestamp;

  // 2. predict imu states
  while (!imu_msgs.empty()) {
    int size = imu_states_sld_win.size();

    auto imu_msg = imu_msgs.front();
    imu_msgs.pop_front();

    if (imu_msg.timestamp == imu_states_sld_win[size - 1].timestamp) {
      continue;
    }

    ImuState imu_state;
    imu_state.timestamp = imu_msg.timestamp;
    imu_state.acc       = imu_msg.linear_acceleration;
    imu_state.gyr       = imu_msg.angular_velocity;
    PredictPoseOfNewImuState(imu_states_sld_win[size - 2], imu_states_sld_win[size - 1], ba, bg, grav, imu_state);

    imu_states_sld_win.push_back(imu_state);

    if (imu_state.timestamp >= sample_states_sld_win.back()->timestamp) {
      // ensure that we have enough imu states
      break;
    }
  }
}

}  // namespace

void LidarOdometry::BuildSldWinLidarResiduals(const std::vector<SurfelCorrespondence> &surfel_corrs, ceres::Problem &problem, std::vector<ceres::ResidualBlockId> &residual_ids) {
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> correlation_matrix;
  correlation_matrix.resize(sample_states_sld_win_.size() - 1, sample_states_sld_win_.size() - 1);
  correlation_matrix.setZero();

  for (const auto &surfel_corr : surfel_corrs) {
    CHECK_LT(surfel_corr.s1->timestamp, surfel_corr.s2->timestamp) << std::fixed << std::setprecision(9) << surfel_corr.s1->timestamp << " " << surfel_corr.s2->timestamp;  // bug: disorder happens

    auto sp1r_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), surfel_corr.s1->timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    CHECK(sp1r_it != sample_states_sld_win_.begin());
    CHECK(sp1r_it != sample_states_sld_win_.end());

    auto sp1l    = *(sp1r_it - 1);
    auto sp1r    = *(sp1r_it);
    auto sp2r_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), surfel_corr.s2->timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    CHECK(sp2r_it != sample_states_sld_win_.begin());
    CHECK(sp2r_it != sample_states_sld_win_.end()) << std::fixed << std::setprecision(9) << surfel_corr.s1->timestamp << " " << surfel_corr.s2->timestamp << " " << sample_states_sld_win_.back()->timestamp;
    auto sp2l = *(sp2r_it - 1);
    auto sp2r = *(sp2r_it);

    auto loss_function = new ceres::CauchyLoss(0.4);  // todo set cauchy loss param
    if (sp1r->timestamp < sp2l->timestamp) {
      auto residual_id = problem.AddResidualBlock(
          new SurfelMatchBinaryFactor4SampleStates(surfel_corr.s1, sp1l, sp1r, surfel_corr.s2, sp2l, sp2r),
          loss_function,
          sp1l->data_cor,
          sp1r->data_cor,
          sp2l->data_cor,
          sp2r->data_cor);
      residual_ids.push_back(residual_id);
    } else if (sp1r == sp2l) {
      auto residual_id = problem.AddResidualBlock(
          new SurfelMatchBinaryFactor3SampleStates(surfel_corr.s1, sp1l, sp1r, surfel_corr.s2, sp2r),
          loss_function,
          sp1l->data_cor,
          sp1r->data_cor,
          sp2r->data_cor);
      residual_ids.push_back(residual_id);
    } else {
      auto residual_id = problem.AddResidualBlock(
          new SurfelMatchBinaryFactor2SampleStates(surfel_corr.s1, sp1l, sp1r, surfel_corr.s2),
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
    CHECK_LT(surfel_corr.s1->timestamp, surfel_corr.s2->timestamp) << std::fixed << std::setprecision(9) << surfel_corr.s1->timestamp << " " << surfel_corr.s2->timestamp;  // bug: disorder happens

    auto sp2r_it = std::upper_bound(sample_states_sld_win_.begin(), sample_states_sld_win_.end(), surfel_corr.s2->timestamp, [](double lhs, const SampleState::Ptr &rhs) { return lhs < rhs->timestamp; });
    CHECK(sp2r_it != sample_states_sld_win_.begin());
    CHECK(sp2r_it != sample_states_sld_win_.end());
    auto sp2l = *(sp2r_it - 1);
    auto sp2r = *(sp2r_it);

    auto loss_function = new ceres::CauchyLoss(0.4);  // todo set cauchy loss param
    auto residual_id   = problem.AddResidualBlock(
        new SurfelMatchUnaryFactor(surfel_corr.s1, surfel_corr.s2, sp2l, sp2r),
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
          sp2->data_cor,
          sp1->data_bias);
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
          sp3->data_cor,
          sp1->data_bias);
      residual_ids.push_back(residual_id);
    }
  }
}

bool LidarOdometry::SyncPackages(MeasureGroup &mg) {
  // clear measure group
  mg = MeasureGroup();

  // check if we have enough imu and lidar messages
  if (imu_buff_.empty() || points_buff_.empty()) {
    return false;
  }

  double common_begin_time = std::max(imu_buff_.front().timestamp, points_buff_.front().time);
  double common_end_time   = std::min(imu_buff_.back().timestamp, points_buff_.back().time);
  if (common_end_time - common_begin_time < config_.sweep_duration + 0.3) {
    return false;
  }

  static bool is_first_sweep_synced = false;
  if (!is_first_sweep_synced) {
    while (imu_buff_.front().timestamp < common_begin_time) {
      imu_buff_.pop_front();
    }
    is_first_sweep_synced = true;
  }

  // add imu and lidar messages to the measure group
  mg.sweep_beg_time = imu_buff_.at(0).timestamp;
  mg.sweep_end_time = imu_buff_.at(config_.imu_num_per_sweep).timestamp;

  for (int i = 0; i <= config_.imu_num_per_sweep; ++i) {
    mg.imu_msgs.push_back(imu_buff_.at(i));
  }
  for (int i = 0; i < points_buff_.size(); ++i) {
    if (points_buff_.at(i).time <= mg.sweep_beg_time) {
      continue;
    }
    if (points_buff_.at(i).time >= mg.sweep_end_time) {
      break;
    }
    mg.lidar_points.push_back(points_buff_.at(i));
  }

  // remove synced imu and lidar messages
  imu_buff_.erase(imu_buff_.begin(), imu_buff_.begin() + config_.imu_num_per_sweep);
  while (!points_buff_.empty() && points_buff_.front().time < mg.sweep_end_time) {
    points_buff_.pop_front();
  }

  static int sweep_id = 0;
  mg.sweep_id         = sweep_id++;
  LOG(INFO) << fmt::format("SyncPackages done: sweep_{}_{}[{:.9f},{:.9f}] imu_{}[{:.9f},{:.9f}] points_{}[{:.9f},{:.9f}]",
                           mg.sweep_id,
                           mg.sweep_end_time - mg.sweep_beg_time,
                           mg.sweep_beg_time,
                           mg.sweep_end_time,
                           mg.imu_msgs.back().timestamp - mg.imu_msgs.front().timestamp,
                           mg.imu_msgs.front().timestamp,
                           mg.imu_msgs.back().timestamp,
                           mg.lidar_points.back().time - mg.lidar_points.front().time,
                           mg.lidar_points.front().time,
                           mg.lidar_points.back().time);

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

  MeasureGroup mg;
  if (!SyncPackages(mg)) {
    return;
  }

  // 1. collect scan to sweep
  const auto &sweep = mg.lidar_points;

  // 2. integrate to get IMU poses in windows and extend sliding window buffer
  PredictImuStatesAndSampleStates(mg, config_, imu_states_sld_win_, sample_states_sld_win_);
  points_buff_sld_win_.insert(points_buff_sld_win_.end(), sweep.begin(), sweep.end());

  // 3. undistort sweep by IMU poses
  std::deque<hilti_ros::Point> sweep_undistorted;
  UndistortSweep(sweep, imu_states_sld_win_, sweep_undistorted);

  // 4. extract surfels and add to windows, the first time surfels will be add to global map
  std::deque<Surfel::Ptr> surfels_sweep;
  GlobalMap               map;
  BuildSurfels(sweep_undistorted, surfels_sweep, map);
  surfels_sld_win_.insert(surfels_sld_win_.end(), surfels_sweep.begin(), surfels_sweep.end());
  UpdateSurfelPoses(imu_states_sld_win_, surfels_sld_win_);
  UpdateSurfelNormals(imu_states_sld_win_, surfels_sweep);

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
    option.num_threads                  = config_.opt_thread_num;
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
    LOG(INFO) << summary.FullReport();

    UpdateImuPoses(sample_states_sld_win_, imu_states_sld_win_);
    UpdateSurfelPoses(imu_states_sld_win_, surfels_sld_win_);
    UpdateSamplePoses(sample_states_sld_win_);

    PrintSurfelResiduals(surfel_sld_win_residual_ids, problem, "Sliding Window");
    PrintSurfelResiduals(surfel_fix_win_residual_ids, problem, "Fixed Window");
    PrintImuResiduals(imu_residual_ids, problem);
    PrintSampleStates(sample_states_sld_win_);
  }

  std::deque<hilti_ros::Point> points_buff_shift_out;
  std::deque<ImuState>         imu_states_shift_out;
  auto                         imu_states_shift_out_full = imu_states_sld_win_;  // here we copy the imu buffer to undistort the shift out points
  ShrinkToFit(
      sample_states_sld_win_,
      imu_states_sld_win_,
      imu_states_shift_out,
      points_buff_sld_win_,
      points_buff_shift_out,
      surfels_sld_win_,
      surfels_fix_win_,
      config_.sliding_window_duration,
      config_.fixed_window_duration);

  for (auto &e : imu_states_shift_out) {
    io_.AddOdom(e.timestamp, e.rot, e.pos);
  }

  PubSurfels(surfels_sld_win_, pub_plane_map_);
  if (!points_buff_shift_out.empty()) {
    sensor_msgs::PointCloud2     msg;
    std::deque<hilti_ros::Point> points_buff_shift_out_world;
    UndistortSweep(points_buff_shift_out, imu_states_shift_out_full, points_buff_shift_out_world);
    pcl::PointCloud<hilti_ros::Point> cloud;
    for (auto &e : points_buff_shift_out_world) {
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
    transform.setOrigin(tf::Vector3(imu_states_sld_win_.back().pos[0], imu_states_sld_win_.back().pos[1], imu_states_sld_win_.back().pos[2]));
    transform.setRotation(tf::Quaternion(imu_states_sld_win_.back().rot.x(), imu_states_sld_win_.back().rot.y(), imu_states_sld_win_.back().rot.z(), imu_states_sld_win_.back().rot.w()));
    br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(imu_states_sld_win_.back().timestamp), "world", "imu_link"));
  }
  if (!imu_states_shift_out.empty()) {
    static nav_msgs::Path path;
    path.header.stamp    = ros::Time().fromSec(imu_states_shift_out.back().timestamp);
    path.header.frame_id = "world";
    for (int i = 0; i < imu_states_shift_out.size(); ++i) {
      geometry_msgs::PoseStamped pose;
      pose.header.stamp       = ros::Time().fromSec(imu_states_shift_out[i].timestamp);
      pose.header.frame_id    = "body";
      pose.pose.position.x    = imu_states_shift_out[i].pos[0];
      pose.pose.position.y    = imu_states_shift_out[i].pos[1];
      pose.pose.position.z    = imu_states_shift_out[i].pos[2];
      pose.pose.orientation.x = imu_states_shift_out[i].rot.x();
      pose.pose.orientation.y = imu_states_shift_out[i].rot.y();
      pose.pose.orientation.z = imu_states_shift_out[i].rot.z();
      pose.pose.orientation.w = imu_states_shift_out[i].rot.w();
      path.poses.push_back(pose);
    }
    pub_imu_path_.publish(path);
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
  pub_imu_path_          = nh_.advertise<nav_msgs::Path>("/imu_path", 10);
}

LidarOdometry::~LidarOdometry() {
  for (auto &e : imu_states_sld_win_) {
    io_.AddOdom(e.timestamp, e.rot, e.pos);
  }
}
