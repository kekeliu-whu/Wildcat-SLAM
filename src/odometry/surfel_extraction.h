#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP

#include <glog/logging.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <string>
#include <unordered_map>

#include "common/common.h"

#define HASH_P 116101
#define MAX_N 10000000000

struct Surfel {
  double           timestamp;
  double           resolution;
  Eigen::Vector3d  center;
  Eigen::Vector3d  norm;
  Eigen::Matrix3cd covariance;
};

typedef struct PointWithCovMeta {
  double          timestamp;
  Eigen::Vector3d pw;  // point in world frame
} pointWithCovMeta;

// 3D point with covariance
typedef struct PointWithCov : pointWithCovMeta {
} pointWithCov;

// a point to plane matching structure
typedef struct PointPlaneMatchInfo {
  pointWithCov                pv;
  Eigen::Vector3d             normal;     // plane normal vector in world frame
  Eigen::Vector3d             center;     // plane center point in world frame
  Eigen::Matrix<double, 6, 6> plane_cov;  // plane covariance in world frame
  double                      d;          // used to compute point-plane distance
  int                         layer;
} ptpl;

typedef struct Plane {
  double          timestamp;
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Vector3d y_normal;
  Eigen::Vector3d x_normal;
  Eigen::Matrix3d covariance;
  float           radius          = 0;  // if the plane points are evenly distributed in a circle, then radius*2 will be real radius of the circle
  float           min_eigen_value = 1;
  float           mid_eigen_value = 1;
  float           max_eigen_value = 1;
  float           d               = 0;
  int             points_size     = 0;

  bool is_plane = false;
  int  id;
} Plane;

class VoxelLoc {
 public:
  int64_t x, y, z;

  VoxelLoc(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VoxelLoc &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <>
struct hash<VoxelLoc> {
  int64_t operator()(const VoxelLoc &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

class OctoTree {
 public:
  std::vector<pointWithCovMeta> temp_points_;  // all points in an octo tree
  std::vector<pointWithCovMeta> new_points_;   // new points in an octo tree
  Plane                        *plane_ptr_;
  int                           max_layer_;
  int                           layer_;
  int                           octo_state_;  // 0 is end of tree, 1 is not
  OctoTree                     *leaves_[8];
  double                        voxel_center_[3];  // x, y, z
  std::vector<int>              layer_point_size_;
  float                         quarter_length_;
  float                         planer_threshold_;
  int                           max_plane_update_threshold_;
  int                           update_size_threshold_;
  int                           all_points_num_;
  int                           new_points_num_;
  int                           max_points_size_;
  int                           max_cov_points_size_;
  bool                          init_octo_;
  bool                          update_cov_enable_;
  bool                          update_enable_;
  double                        min_plane_likeness_;

  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           int max_point_size, int max_cov_points_size, float planer_threshold, double min_plane_likeness);

  // check is plane , calc plane parameters including plane covariance
  void InitPlane(const std::vector<pointWithCovMeta> &points, Plane *plane);

  // only update plane normal, center and radius with new points
  void UpdatePlane(const std::vector<pointWithCovMeta> &points, Plane *plane);

  void InitOctoTree();

  void CutOctoTree();

  void UpdateOctoTree(const pointWithCovMeta &pv);

  void ExtractSurfelInfo(std::vector<pcl::PointCloud<PointType>> &cloud_multi_layers, std::vector<Surfel> &surfels, int cur_layer = 0);
};

void BuildVoxelMap(const std::vector<pointWithCovMeta> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float                               planer_threshold,
                   double                                    min_plane_likeness,
                   std::unordered_map<VoxelLoc, OctoTree *> &feat_map);

void BuildSurfels(const pcl::PointCloud<hilti_ros::Point>::Ptr &cloud,
                  std::vector<Surfel>                          &surfels);

#endif