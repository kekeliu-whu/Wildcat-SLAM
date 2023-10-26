#ifndef MSF_LOAM_VELODYNE_MSG_CONVERSION_H
#define MSF_LOAM_VELODYNE_MSG_CONVERSION_H

#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/Vector3.h>

#include "common/common.h"
#include "common/rigid_transform.h"
#include "common/time.h"

Vector3d FromROS(const geometry_msgs::Vector3_<std::allocator<void>> &v);

Vector3d FromROS(const geometry_msgs::Point_<std::allocator<void>> &o);

Quaterniond FromROS(const geometry_msgs::Quaternion_<std::allocator<void>> &p);

Rigid3d FromROS(const geometry_msgs::PoseWithCovariance &pose_msg);

geometry_msgs::PoseWithCovariance ToROS(const Rigid3d &pose);

Time FromROS(const ros::Time &time);

ros::Time ToROS(const Time &time);

#endif  // MSF_LOAM_VELODYNE_MSG_CONVERSION_H
