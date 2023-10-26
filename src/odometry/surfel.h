#pragma once

#include <glog/logging.h>
#include <Eigen/Dense>
#include <memory>

#include "common/common.h"

struct SampleState {
  typedef std::shared_ptr<SampleState> Ptr;

  double               timestamp;
  double               data_cor[12] = {0};  // q, t, bg, ba
  Eigen::Map<Vector3d> rot_cor{data_cor + 0};
  Eigen::Map<Vector3d> pos_cor{data_cor + 3};
  Eigen::Map<Vector3d> bg{data_cor + 6};
  Eigen::Map<Vector3d> ba{data_cor + 9};

  Vector3d grav;

  Quaterniond rot;
  Vector3d    pos;
};

struct ImuState {
  typedef std::shared_ptr<ImuState> Ptr;

  double      timestamp;
  Vector3d    pos;
  Quaterniond rot;
  Vector3d    acc;
  Vector3d    gyr;
};

struct Surfel {
  typedef std::shared_ptr<Surfel> Ptr;

 public:
  Surfel(double timestamp, const Vector3d &center, const Matrix3d &covariance, const Vector3d &norm, double resolution, double plane_std_deviation) : timestamp(timestamp), center(center), covariance(covariance), norm(norm), resolution(resolution), plane_std_deviation(plane_std_deviation) {
  }

  /**
   * @brief Update the pose of the surfel
   *
   * @param pos
   * @param rot
   */
  void UpdatePose(const Vector3d &pos, const Quaterniond &rot) {
    this->pos = pos;
    this->rot = rot;

    if (!is_in_body_frame) {
      is_in_body_frame = true;
      center           = rot.conjugate() * (center - pos);
      norm             = rot.conjugate() * norm;
      covariance       = rot.conjugate() * covariance * rot;
    }
  }

  /**
   * @brief Get the surfel center in world frame
   *
   * Used in knn search and surfel visualization
   *
   * @return Vector3d
   */
  Vector3d GetCenterInWorld() const {
    return rot * center + pos;
  }

  /**
   * @brief Get the surfel normal in world frame
   *
   * Used in knn search and surfel visualization
   *
   * @return Vector3d
   */
  Vector3d GetNormInWorld() const {
    return rot * norm;
  }

  /**
   * @brief Get the covariance in world frame
   *
   * Used in surfel match weight and surfel visualization
   *
   * @return Matrix3d
   */
  Matrix3d GetCovarianceInWorld() const {
    return rot * covariance * rot.conjugate();
  }

  /**
   * @brief Get the surfel center in body frame
   *
   * Used in surfel match factor
   *
   * @return Vector3d
   */
  Vector3d CenterInBody() const {
    CHECK(is_in_body_frame) << "You should call UpdatePose first.";
    return center;
  }

  double AngularDistance(const Surfel &surfel) const {
    return std::acos(GetNormInWorld().dot(surfel.GetNormInWorld()));
  }

 public:
  double timestamp;
  double resolution;
  double plane_std_deviation;

  Quaterniond rot{1, 0, 0, 0};  // body frame to world frame
  Vector3d    pos{0, 0, 0};     // body frame to world frame

 private:
  bool     is_in_body_frame = false;
  Vector3d center;      // in body frame
  Matrix3d covariance;  // in body frame
  Vector3d norm;        // in body frame
};

struct SurfelCorrespondence {
  Surfel::Ptr s1;
  Surfel::Ptr s2;
};
