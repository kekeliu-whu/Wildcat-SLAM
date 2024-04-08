#pragma once

#include <glog/logging.h>
#include <Eigen/Dense>
#include <memory>

#include "common/common.h"

struct SampleState {
  typedef std::shared_ptr<SampleState> Ptr;

  double               timestamp;
  double               data_cor[6] = {0};  // q, t, bg, ba
  Eigen::Map<Vector3d> rot_cor{data_cor + 0};
  Eigen::Map<Vector3d> pos_cor{data_cor + 3};
  Eigen::Map<Vector3d> bg{data_bias + 0};
  Eigen::Map<Vector3d> ba{data_bias + 3};

  static double data_bias[6];

  Vector3d grav;
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

  double   timestamp;
  Vector3d center;      // in world frame
  Vector3d normal;      // in world frame
  Matrix3d covariance;  // in world frame
  double   resolution;

 public:
  Surfel(double timestamp, const Vector3d &center, const Matrix3d &covariance, const Vector3d &norm, double resolution) : timestamp(timestamp), center(center), covariance(covariance), normal(norm), resolution(resolution) {
  }

  double AngularDistance(const Surfel &surfel) const {
    return std::acos(normal.dot(surfel.normal));
  }

  /**
   * @brief Update the pose of the surfel
   *
   * @param pos
   * @param rot
   */
  void UpdatePose(const Vector3d &pos_new, const Quaterniond &rot_new) {
    if (!pose_inited) {
      this->pos   = pos_new;
      this->rot   = rot_new;
      pose_inited = true;
      return;
    }

    // T_new * T_old^-1
    Quaterniond rot_cor = rot_new * this->rot.conjugate();
    Vector3d    pos_cor = pos_new - rot_new * this->rot.conjugate() * this->pos;

    center     = rot_cor * center + pos_cor;
    normal     = rot_cor * normal + pos_cor;
    covariance = rot_cor * covariance * rot_cor.conjugate();

    this->pos = pos_new;
    this->rot = rot_new;
  }

 private:
  bool        pose_inited = false;
  Quaterniond rot;  // body frame to world frame
  Vector3d    pos;  // body frame to world frame
};

struct SurfelCorrespondence {
  Surfel::Ptr s1;
  Surfel::Ptr s2;
};
