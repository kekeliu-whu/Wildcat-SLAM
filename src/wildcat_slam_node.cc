
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <signal.h>

#include "common/msg_conversion.h"
#include "odometry/surfel_odometry.h"

DEFINE_bool(is_offline_mode, true, "Runtime mode: online or offline.");

DEFINE_string(bag_filename, "/home/rick/Documents/raw_data/hilti/exp04_construction_upper_level.bag-filtered.bag", "Bag file to read in offline mode.");

volatile sig_atomic_t g_signal_stop = 0;

void signal_handler(int status) {
  g_signal_stop = 1;
}

// DEFINE_string(config_filename, "", "Configuration file.");

void HandleImuMessage(
    const sensor_msgs::ImuConstPtr        &msg,
    const std::shared_ptr<SurfelOdometry> &laser_odometry_handler) {
  ImuData imu_data;
  imu_data.time                = FromROS(msg->header.stamp);
  imu_data.linear_acceleration = FromROS(msg->linear_acceleration);
  imu_data.angular_velocity    = FromROS(msg->angular_velocity);
  laser_odometry_handler->AddImuData(imu_data);
}

void HandleLidarMessage(
    const sensor_msgs::PointCloud2ConstPtr &msg,
    const std::shared_ptr<SurfelOdometry>  &laser_odometry_handler) {
  pcl::PointCloud<hilti_ros::Point> cloud;
  pcl::fromROSMsg(*msg, cloud);
  laser_odometry_handler->AddLidarPoints();
}

int main(int argc, char **argv) {
  // Set glog and gflags
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Set ROS node
  ros::init(argc, argv, "nsf_loam_node");
  ros::NodeHandle nh;

  signal(SIGINT, signal_handler);

  CHECK_EQ(FLAGS_is_offline_mode, true);
  CHECK_NE(FLAGS_bag_filename, "");

  auto so = std::make_shared<SurfelOdometry>(0.05, 800, 10);

  if (FLAGS_is_offline_mode) {
    LOG(INFO) << "Using offline mode ...";
    rosbag::Bag bag;
    bag.open(FLAGS_bag_filename);
    LOG(INFO) << "Reading bag file " << FLAGS_bag_filename << " ...";
    for (auto &m : rosbag::View(bag)) {
      if (g_signal_stop) {
        LOG(INFO) << "Exit.";
        break;
      }
      if (m.isType<sensor_msgs::PointCloud2>()) {
        HandleLidarMessage(m.instantiate<sensor_msgs::PointCloud2>(),
                           so);
      } else if (m.isType<sensor_msgs::Imu>()) {
        HandleImuMessage(m.instantiate<sensor_msgs::Imu>(),
                         so);
      }
    }
    bag.close();
  }

  return 0;
}
