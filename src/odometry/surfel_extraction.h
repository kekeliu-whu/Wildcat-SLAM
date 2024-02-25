#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <pcl/common/io.h>
#include <ros/publisher.h>
#include <rosbag/bag.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "common/common.h"
#include "odometry/surfel.h"

#define HASH_P 116101
#define MAX_N 10000000000

// 3D point with covariance
struct PointWithCov {
  double   timestamp;
  Vector3d pw;  // point in world frame
};

// a point to plane matching structure
typedef struct PointPlaneMatchInfo {
  PointWithCov                pv;
  Vector3d                    normal;     // plane normal vector in world frame
  Vector3d                    center;     // plane center point in world frame
  Eigen::Matrix<double, 6, 6> plane_cov;  // plane covariance in world frame
  double                      d;          // used to compute point-plane distance
  int                         layer;
} ptpl;

typedef struct Plane {
  double   timestamp;
  Vector3d center;
  Vector3d normal;
  Vector3d y_normal;
  Vector3d x_normal;
  Matrix3d covariance;
  float    radius          = 0;  // if the plane points are evenly distributed in a circle, then radius*2 will be real radius of the circle
  float    min_eigen_value = 1;
  float    mid_eigen_value = 1;
  float    max_eigen_value = 1;
  float    d               = 0;
  int      points_size     = 0;

  bool is_plane = false;
  int  id;
} Plane;

class VoxelLoc {
 public:
  int32_t x, y, z;

  VoxelLoc(const Vector3d &pos, double resolution) {
    Eigen::Vector3i loc = (pos / resolution).array().floor().cast<int32_t>();
    x                   = loc[0];
    y                   = loc[1];
    z                   = loc[2];
  }

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
  std::vector<PointWithCov> temp_points_;  // all points in an octo tree
  Plane                    *plane_ptr_;
  int                       layer_;
  int                       octo_state_;  // 0 is leaf node, 1 is not
  OctoTree                 *leaves_[8];
  double                    voxel_center_[3];
  float                     quarter_length_;
  Vector3d                  view_point_;  // todo set plane norm by trajectory instead of one point

  float  planer_threshold_;
  double min_plane_likeness_;

  int              layer_point_size_plane_threshold_;
  std::vector<int> layer_point_size_;
  int              max_layer_;

  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           float planer_threshold, double min_plane_likeness,
           const Vector3d &view_point);

  ~OctoTree() {
    delete plane_ptr_;

    for (int i = 0; i < 8; ++i) {
      if (leaves_[i]) {
        delete leaves_[i];
      }
    }
  }

  // check is plane , calc plane parameters including plane covariance
  void InitPlane(const std::vector<PointWithCov> &points, Plane *plane);

  void InitOctoTree();

  void CutOctoTree();

  void ExtractSurfelInfo(std::deque<Surfel::Ptr> &surfels, int cur_layer = 0);
};

class GlobalMap {
 public:
  absl::flat_hash_map<VoxelLoc, OctoTree *> feat_map;

  ~GlobalMap() {
    for (auto &e : feat_map) {
      delete e.second;
    }
  }
};

void BuildVoxelMap(const std::vector<PointWithCov>           &input_points,
                   const Vector3d                            &view_point,
                   const float                                voxel_size,
                   const int                                  max_layer,
                   const std::vector<int>                    &layer_point_size,
                   const float                                planer_threshold,
                   double                                     min_plane_likeness,
                   absl::flat_hash_map<VoxelLoc, OctoTree *> &feat_map);

void BuildSurfels(const std::vector<hilti_ros::Point> &cloud,
                  std::deque<Surfel::Ptr>             &surfels,
                  GlobalMap                           &map);

void PubPlaneMap(const absl::flat_hash_map<VoxelLoc, OctoTree *> &feat_map,
                 const ros::Publisher                            &plane_map_pub);

void PubSurfels(std::deque<Surfel::Ptr> surfels,
                const ros::Publisher   &plane_map_pub);

#endif