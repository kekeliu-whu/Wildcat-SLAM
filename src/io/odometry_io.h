#pragma once

#include <Eigen/Eigen>
#include <fstream>
#include <iomanip>
#include <string>

#include "glog/logging.h"

class OdometryIO {
 public:
  OdometryIO(const std::string& path) : result_path_(path) {}

  void AddOdom(double timestamp, const Eigen::Quaterniond& rot, const Eigen::Vector3d& pos) {
    if (!ofs_) {
      ofs_.open(result_path_ + "/trajectory.txt");
      ofs_ << "#timestamp_s tx ty tz qx qy qz qw" << std::endl;
    }
    ofs_ << std::fixed << std::setprecision(9) << timestamp << " " << pos.x() << " " << pos.y() << " " << pos.z() << " " << rot.x() << " " << rot.y() << " " << rot.z() << " " << rot.w() << std::endl;
  }

  ~OdometryIO() {
    ofs_.close();
    LOG(INFO) << "Odometry saved to " << result_path_ << "/trajectory.txt";
  }

 private:
  std::ofstream ofs_;
  std::string   result_path_;
};
