#include <pcl/io/ply_io.h>

#include "absl/container/flat_hash_map.h"
#include "ros/publisher.h"
#include "ros/rate.h"
#include "surfel_extraction.h"

namespace {

int g_plane_id = 0;

void ClusterSurfels(
    const std::vector<PointWithCov> &points,
    double                           resolution,
    const Vector3d                  &view_point,
    double                           planer_threshold,
    double                           min_plane_likeness,
    std::deque<Surfel::Ptr>         &surfels) {
  // 1. cluster points
  std::vector<std::vector<PointWithCov>> cluster_points;
  cluster_points.push_back({points[0]});
  for (auto i = 1; i < points.size(); ++i) {
    // todo magic number
    if (points[i].timestamp - cluster_points.back().back().timestamp > 0.05) {
      cluster_points.push_back({points[i]});
    } else {
      cluster_points.back().push_back(points[i]);
    }
  }

  // 2. extract surfels from cluster
  for (auto &cluster : cluster_points) {
    if (cluster.size() < 20) {
      continue;
    }
    Vector3d center      = Vector3d::Zero();
    Matrix3d covariance  = Matrix3d::Zero();
    double   timestamp   = 0;
    int      points_size = cluster.size();
    for (auto pv : cluster) {
      covariance += pv.pw * pv.pw.transpose();
      center += pv.pw;
      timestamp += pv.timestamp;
    }
    center     = center / points_size;
    timestamp  = timestamp / points_size;
    covariance = covariance / points_size - center * center.transpose();

    Eigen::SelfAdjointEigenSolver<Matrix3d> es(covariance);
    Matrix3d                                evecs    = es.eigenvectors().real();
    Vector3d                                evals    = es.eigenvalues().real();
    Eigen::Matrix3f::Index                  evalsMin = 0, evalsMid = 1, evalsMax = 2;  // SelfAdjointEigenSolver's eigen values are in increase order
    double                                  plane_likeness = 2 * (evals(evalsMid) - evals(evalsMin)) / evals.sum();
    if (evals(evalsMin) > planer_threshold || plane_likeness < min_plane_likeness) {
      continue;
    }

    Vector3d norm = evecs.col(evalsMin);
    if (norm.dot(center - view_point) < 0) {
      norm = -norm;
    }
    Surfel::Ptr sf{new Surfel(timestamp, center, covariance, norm, resolution)};
    surfels.push_back(sf);
  }
}

}  // namespace

OctoTree::OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
                   float planer_threshold, double min_plane_likeness, const Vector3d &view_point)
    : max_layer_(max_layer), layer_(layer), layer_point_size_(layer_point_size), planer_threshold_(planer_threshold), min_plane_likeness_(min_plane_likeness), view_point_(view_point) {
  temp_points_.clear();
  octo_state_                       = 0;
  layer_point_size_plane_threshold_ = layer_point_size_[layer_];
  for (int i = 0; i < 8; i++) {
    leaves_[i] = nullptr;
  }
  plane_ptr_ = new Plane;
}

// check is plane , calc plane parameters including plane covariance
void OctoTree::InitPlane(const std::vector<PointWithCov> &points, Plane *plane) {
  plane->covariance  = Matrix3d::Zero();
  plane->center      = Vector3d::Zero();
  plane->normal      = Vector3d::Zero();
  plane->points_size = points.size();
  plane->radius      = 0;
  plane->timestamp   = 0;
  for (auto pv : points) {
    plane->covariance += pv.pw * pv.pw.transpose();
    plane->center += pv.pw;
    plane->timestamp += pv.timestamp;
  }
  plane->center     = plane->center / plane->points_size;
  plane->timestamp  = plane->timestamp / plane->points_size;
  plane->covariance = plane->covariance / plane->points_size -
                      plane->center * plane->center.transpose();
  Eigen::SelfAdjointEigenSolver<Matrix3d> es(plane->covariance);
  Matrix3d                                evecs    = es.eigenvectors().real();
  Vector3d                                evals    = es.eigenvalues().real();
  Eigen::Matrix3f::Index                  evalsMin = 0, evalsMid = 1, evalsMax = 2;  // SelfAdjointEigenSolver's eigen values are in increase order
  // plane covariance calculation
  Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
      1.0 / plane->points_size;
  double plane_likeness = 2 * (evals(evalsMid) - evals(evalsMin)) / evals.sum();
  if (evals(evalsMin) < planer_threshold_ && plane_likeness > min_plane_likeness_) {
    plane->is_plane = true;
  } else {
    plane->is_plane = false;
  }

  plane->normal = evecs.col(evalsMin);
  if (plane->normal.dot(plane->center - view_point_) < 0) {
    plane->normal = -plane->normal;
  }
  plane->y_normal        = evecs.col(evalsMid);
  plane->x_normal        = evecs.col(evalsMax);
  plane->min_eigen_value = evals(evalsMin);
  plane->mid_eigen_value = evals(evalsMid);
  plane->max_eigen_value = evals(evalsMax);
  plane->radius          = sqrt(evals(evalsMax));
  plane->d               = -plane->normal.dot(plane->center);

  plane->id = g_plane_id++;
}

void OctoTree::InitOctoTree() {
  if (temp_points_.size() > layer_point_size_plane_threshold_) {
    InitPlane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane == true) {
      octo_state_ = 0;
      // note by kk: here we force to split voxel
      CutOctoTree();
    } else {
      octo_state_ = 1;
      CutOctoTree();
    }
  }
}

void OctoTree::CutOctoTree() {
  if (layer_ >= max_layer_) {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++) {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].pw[0] > voxel_center_[0]) {
      xyz[0] = 1;
    }
    if (temp_points_[i].pw[1] > voxel_center_[1]) {
      xyz[1] = 1;
    }
    if (temp_points_[i].pw[2] > voxel_center_[2]) {
      xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr) {
      leaves_[leafnum] = new OctoTree(
          max_layer_, layer_ + 1, layer_point_size_,
          planer_threshold_, min_plane_likeness_, view_point_);
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quarter_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quarter_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quarter_length_;
      leaves_[leafnum]->quarter_length_  = quarter_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
  }
  for (uint i = 0; i < 8; i++) {
    if (leaves_[i] != nullptr) {
      if (leaves_[i]->temp_points_.size() >
          leaves_[i]->layer_point_size_plane_threshold_) {
        InitPlane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane) {
          leaves_[i]->octo_state_ = 0;
        } else {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->CutOctoTree();
        }
      }
    }
  }
}

void BuildVoxelMap(const std::vector<PointWithCov>           &input_points,
                   const Vector3d                            &view_point,
                   const float                                voxel_size,
                   const int                                  max_layer,
                   const std::vector<int>                    &layer_point_size,
                   const float                                planer_threshold,
                   double                                     min_plane_likeness,
                   absl::flat_hash_map<VoxelLoc, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const PointWithCov p_v = input_points[i];
    // 1. compute voxel position
    VoxelLoc position{p_v.pw, voxel_size};
    auto     iter = feat_map.find(position);
    // 2. put point into voxel
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size,
                       planer_threshold, min_plane_likeness, view_point);
      feat_map[position]                   = octo_tree;
      feat_map[position]->quarter_length_  = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
  }
  // 3. init octo tree
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->InitOctoTree();
  }
}

struct M_POINT {
  Vector3d              center;
  int                   count = 0;
  std::vector<Vector3d> points;
};

template <typename T>
void DownSamplingVoxel(const pcl::PointCloud<PointType> &cloud_in,
                       pcl::PointCloud<PointType>       &cloud_out,
                       double                            voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  absl::flat_hash_map<VoxelLoc, M_POINT> feat_map;

  for (uint i = 0; i < cloud_in.size(); i++) {
    Vector3d p_c = cloud_in[i].getVector3fMap().cast<double>();
    VoxelLoc position(p_c, voxel_size);
    auto     iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.center += p_c;
      iter->second.count++;
    } else {
      M_POINT p;
      p.center           = p_c;
      p.count            = 1;
      feat_map[position] = p;
    }
  }

  cloud_out.clear();
  cloud_out.resize(feat_map.size());

  uint i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    cloud_out[i].getVector3fMap() = iter->second.center.cast<float>() / iter->second.count;
    i++;
  }
}

template <typename T>
void DownSamplingVoxelRandom(const pcl::PointCloud<PointType> &cloud_in,
                             pcl::PointCloud<PointType>       &cloud_out,
                             double                            voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  absl::flat_hash_map<VoxelLoc, M_POINT> feat_map;

  for (uint i = 0; i < cloud_in.size(); i++) {
    Vector3d p_c = cloud_in[i].getVector3fMap().cast<double>();
    VoxelLoc position(p_c, voxel_size);
    auto     iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.points.push_back(p_c);
    } else {
      M_POINT p;
      p.points.push_back(p_c);
      feat_map[position] = p;
    }
  }

  cloud_out.clear();
  cloud_out.resize(feat_map.size());

  uint i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    cloud_out[i].getVector3fMap() = iter->second.points.at(rand() % iter->second.points.size()).cast<float>();
    i++;
  }
}

template void DownSamplingVoxel<PointType>(const pcl::PointCloud<PointType> &cloud_in,
                                           pcl::PointCloud<PointType>       &cloud_out,
                                           double                            voxel_size);

template void DownSamplingVoxelRandom<PointType>(const pcl::PointCloud<PointType> &cloud_in,
                                                 pcl::PointCloud<PointType>       &cloud_out,
                                                 double                            voxel_size);

void OctoTree::ExtractSurfelInfo(std::deque<Surfel::Ptr> &surfels, int cur_layer) {
  if (this->plane_ptr_ && this->plane_ptr_->is_plane) {
    CHECK(!this->temp_points_.empty());
    ClusterSurfels(this->temp_points_, this->quarter_length_ * 4, view_point_, planer_threshold_, min_plane_likeness_, surfels);
  }
  for (auto &leaf : this->leaves_) {
    if (leaf) {
      leaf->ExtractSurfelInfo(surfels, cur_layer + 1);
    }
  }
}

void BuildSurfels(const std::deque<hilti_ros::Point> &cloud, std::deque<Surfel::Ptr> &surfels, GlobalMap &map) {
  std::vector<PointWithCov> points;

  for (auto &e : cloud) {
    PointWithCov np;
    np.timestamp = e.time;
    np.pw        = e.getVector3fMap().cast<double>();
    points.push_back(np);
  }

  // todo magic number
  BuildVoxelMap(points, Vector3d::Zero(), 0.8, 2, {20, 20, 20, 20}, 0.01, 0.1, map.feat_map);

  std::vector<pcl::PointCloud<PointType>> cloud_surfel_multi_layers(4);
  for (auto &e : map.feat_map) {
    e.second->ExtractSurfelInfo(surfels);
  }

  std::sort(surfels.begin(), surfels.end(), [](const auto &a, const auto &b) { return a->timestamp < b->timestamp; });

  LOG(INFO) << "Surfel Extraction done, surfel count = " << surfels.size();
}

// Local function to force the axis to be right handed for 3D. Taken from ecl_statistics
void makeRightHanded(Matrix3d &eigenvectors, Vector3d &eigenvalues) {
  // Note that sorting of eigenvalues may end up with left-hand coordinate system.
  // So here we correctly sort it so that it does end up being righ-handed and normalised.
  Vector3d c0 = eigenvectors.block<3, 1>(0, 0);
  c0.normalize();
  Vector3d c1 = eigenvectors.block<3, 1>(0, 1);
  c1.normalize();
  Vector3d c2 = eigenvectors.block<3, 1>(0, 2);
  c2.normalize();
  Vector3d cc = c0.cross(c1);
  if (cc.dot(c2) < 0) {
    eigenvectors << c1, c0, c2;
    double e       = eigenvalues[0];
    eigenvalues[0] = eigenvalues[1];
    eigenvalues[1] = e;
  } else {
    eigenvectors << c0, c1, c2;
  }
}

void PubSurfels(std::deque<Surfel::Ptr> surfels,
                const ros::Publisher   &plane_map_pub) {
  visualization_msgs::MarkerArray voxel_planes;

  for (auto &surfel : surfels) {
    Vector3d eigenvalues(Vector3d::Identity());
    Matrix3d eigenvectors(Matrix3d::Zero());

    // NOTE: The SelfAdjointEigenSolver only references the lower triangular part of the covariance matrix
    // FIXME: Should we use Eigen's pseudoEigenvectors() ?
    Eigen::SelfAdjointEigenSolver<Matrix3d> eigensolver(surfel->covariance);
    // Compute eigenvectors and eigenvalues
    if (eigensolver.info() == Eigen::Success) {
      eigenvalues  = eigensolver.eigenvalues();
      eigenvectors = eigensolver.eigenvectors();
    } else {
      ROS_WARN_THROTTLE(1, "failed to compute eigen vectors/values for position. Is the covariance matrix correct?");
      eigenvalues  = Vector3d::Zero();  // Setting the scale to zero will hide it on the screen
      eigenvectors = Matrix3d::Identity();
    }

    // Be sure we have a right-handed orientation system
    makeRightHanded(eigenvectors, eigenvalues);

    // Define the rotation
    Matrix3d rot;
    rot << eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2),
        eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2),
        eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2);
    Quaterniond qq{rot};

    auto center = surfel->center;
    auto norm   = surfel->normal;

    static int                 id = 0;
    visualization_msgs::Marker plane;
    plane.header.frame_id = "world";
    plane.header.stamp    = ros::Time();
    plane.ns              = "plane";
    plane.id              = ++id;
    plane.type            = visualization_msgs::Marker::SPHERE;
    plane.action          = visualization_msgs::Marker::ADD;
    plane.pose.position.x = center[0];
    plane.pose.position.y = center[1];
    plane.pose.position.z = center[2];
    geometry_msgs::Quaternion q;
    q.w                    = qq.w();
    q.x                    = qq.x();
    q.y                    = qq.y();
    q.z                    = qq.z();
    plane.pose.orientation = q;
    plane.scale.x          = 3 * sqrt(eigenvalues[0]);
    plane.scale.y          = 3 * sqrt(eigenvalues[1]);
    plane.scale.z          = 3 * sqrt(eigenvalues[2]);
    plane.color.a          = 1;
    plane.color.r          = (norm[0] + 1) / 2;
    plane.color.g          = (norm[1] + 1) / 2;
    plane.color.b          = (norm[2] + 1) / 2;
    plane.lifetime         = ros::Duration();
    voxel_planes.markers.push_back(plane);
  }

  {
    // delete all history markers
    auto marker_array_msg = visualization_msgs::MarkerArray();
    auto marker           = visualization_msgs::Marker();
    marker.id             = 0;
    marker.ns             = "plane";
    marker.action         = visualization_msgs::Marker::DELETEALL;
    marker_array_msg.markers.push_back(marker);
    plane_map_pub.publish(marker_array_msg);
  }

  plane_map_pub.publish(voxel_planes);
}
