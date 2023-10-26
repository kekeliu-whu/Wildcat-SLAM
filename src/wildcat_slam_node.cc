
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <signal.h>
#include <thread>

#include "common/msg_conversion.h"
#include "odometry/lidar_odometry.h"
#include "sensor/imu_resampler.h"

DEFINE_bool(enable_online_mode, false, "Enable online mode.");
DEFINE_string(bag_filename, "/home/rick/Documents/raw_data/hilti/exp04_construction_upper_level.bag-filtered.bag", "Bag file to read in offline mode.");
DEFINE_int32(imu_rate, 200, "IMU rate in Hz.");

volatile sig_atomic_t         g_signal_stop = 0;
std::shared_ptr<ImuResampler> g_imu_resampler;

void signal_handler(int status) {
  g_signal_stop = 1;
}

// DEFINE_string(config_filename, "", "Configuration file.");

void HandleImuMessage(
    const sensor_msgs::ImuConstPtr       &msg,
    const std::shared_ptr<LidarOdometry> &laser_odometry_handler) {
  ImuData imu_data;
  imu_data.timestamp           = msg->header.stamp.toSec();
  imu_data.linear_acceleration = FromROS(msg->linear_acceleration);
  imu_data.angular_velocity    = FromROS(msg->angular_velocity);

  g_imu_resampler->AddImuData(imu_data);
  auto resampled_imu_data = g_imu_resampler->AdvanceGetResampledImuData();
  if (resampled_imu_data) {
    // LOG(INFO) << "Resampled imu data: " << std::fixed << std::setprecision(6) << resampled_imu_data->timestamp;
    laser_odometry_handler->AddImuData(*resampled_imu_data);
  }
}

void HandleLidarMessage(
    const sensor_msgs::PointCloud2ConstPtr &msg,
    const std::shared_ptr<LidarOdometry>   &laser_odometry_handler) {
  pcl::PointCloud<hilti_ros::Point>::Ptr cloud(new pcl::PointCloud<hilti_ros::Point>);
  pcl::fromROSMsg(*msg, *cloud);
  laser_odometry_handler->AddLidarScan(cloud);
}

int main(int argc, char **argv) {
  // Set glog and gflags
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Set ROS node
  ros::init(argc, argv, "wildcat_slam_node");
  ros::NodeHandle nh;

  signal(SIGINT, signal_handler);

  std::shared_ptr<LidarOdometry> so{new LidarOdometry()};

  g_imu_resampler.reset(new ImuResampler(FLAGS_imu_rate));
  if (FLAGS_enable_online_mode) {
    LOG(INFO) << "Using online mode ...";
    auto imu_sub   = nh.subscribe<sensor_msgs::Imu>("/alphasense/imu", 10000, boost::bind(HandleImuMessage, _1, so));
    auto lidar_sub = nh.subscribe<sensor_msgs::PointCloud2>("/hesai/pandar", 10000, boost::bind(HandleLidarMessage, _1, so));

    while (ros::ok() && !g_signal_stop) {
      ros::spinOnce();
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    LOG(INFO) << "Exit.";
  } else {
    LOG(INFO) << "Using offline mode ...";
    CHECK_NE(FLAGS_bag_filename, "");
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
