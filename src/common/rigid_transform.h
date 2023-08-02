/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOAM_VELODYNE_RIGID_TRANSFORMER_H
#define LOAM_VELODYNE_RIGID_TRANSFORMER_H

#include <cmath>
#include <iostream>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"

template <typename T>
using Vector = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vector7 = Eigen::Matrix<T, 7, 1>;

template <typename T>
using Quaternion = Eigen::Quaternion<T>;

template <typename FloatType>
class Rigid3 {
 public:
  using Vector_     = Vector<FloatType>;
  using Vector7_    = Vector7<FloatType>;
  using Quaternion_ = Quaternion<FloatType>;

  Rigid3()
      : translation_(Vector_::Zero()), rotation_(Quaternion_::Identity()) {}
  Rigid3(const Vector_& translation, const Quaternion_& rotation)
      : translation_(translation), rotation_(rotation) {}
  Rigid3(const Vector7_& vector)
      : translation_(vector.template block<3, 1>(0, 0)),
        rotation_(vector.template block<4, 1>(3, 0)) {}

  static Rigid3 Rotation(const Quaternion_& rotation) {
    return Rigid3(Vector_::Zero(), rotation);
  }

  static Rigid3 Translation(const Vector_& vector) {
    return Rigid3(vector, Quaternion_::Identity());
  }

  Vector7_ ToVector7() {
    Vector7_ vector7;
    vector7.template block<3, 1>(0, 0) = translation_;
    vector7.template block<4, 1>(3, 0) = rotation_.coeffs();
    return vector7;
  }

  static Rigid3<FloatType> Identity() { return Rigid3<FloatType>(); }

  template <typename OtherType>
  Rigid3<OtherType> cast() const {
    return Rigid3<OtherType>(translation_.template cast<OtherType>(),
                             rotation_.template cast<OtherType>());
  }

  const Vector_&     translation() const { return translation_; }
  const Quaternion_& rotation() const { return rotation_; }
  Vector_&           translation() { return translation_; }
  Quaternion_&       rotation() { return rotation_; }

  Rigid3 inverse() const {
    const Quaternion_ rotation    = rotation_.conjugate();
    const Vector_     translation = -(rotation * translation_);
    return Rigid3(translation, rotation);
  }

  std::string DebugString() const {
    std::stringstream ss;
    ss << "{ t: [" << translation().x() << ", " << translation().y() << ", "
       << translation().z() << "], q: [" << rotation().w() << ", "
       << rotation().x() << ", " << rotation().y() << ", " << rotation().z()
       << "] }";
    return ss.str();
  }

  bool IsValid() const {
    return !std::isnan(translation_.x()) && !std::isnan(translation_.y()) &&
           !std::isnan(translation_.z()) &&
           std::abs(FloatType(1) - rotation_.norm()) < FloatType(1e-3);
  }

 private:
  Vector_     translation_;
  Quaternion_ rotation_;
};

template <typename FloatType>
Rigid3<FloatType> operator*(const Rigid3<FloatType>& lhs,
                            const Rigid3<FloatType>& rhs) {
  return Rigid3<FloatType>(
      lhs.rotation() * rhs.translation() + lhs.translation(),
      (lhs.rotation() * rhs.rotation()).normalized());
}

template <typename FloatType>
inline typename Rigid3<FloatType>::Vector_ operator*(
    const Rigid3<FloatType>&                   rigid,
    const typename Rigid3<FloatType>::Vector_& point) {
  return rigid.rotation() * point + rigid.translation();
}

// This is needed for gmock.
template <typename T>
std::ostream& operator<<(std::ostream& os, const Rigid3<T>& rigid) {
  os << rigid.DebugString();
  return os;
}

using Rigid3d = Rigid3<double>;
using Rigid3f = Rigid3<float>;

#include "common/common.h"
#include <pcl/point_cloud.h>

template <typename T>
inline T TransformPoint(const Rigid3d& transform, const T& point) {
  T point_out = point;
  point_out.getVector3fMap() =
      (transform * point.getVector3fMap().template cast<double>()).template cast<float>();
  return point_out;
}

template <typename T>
void TransformPointCloudInPlace(const Rigid3d& transform, pcl::PointCloud<T>& pc) {
  for (auto&& point : pc) {
    point.getVector3fMap() = (transform * point.getVector3fMap().template cast<double>()).template cast<float>();
  }
}

#endif  // LOAM_VELODYNE_RIGID_TRANSFORM_H
