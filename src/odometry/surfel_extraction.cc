#include <pcl/io/ply_io.h>

#include "surfel_extraction.h"

namespace {

int g_plane_id = 0;

}  // namespace

OctoTree::OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
                   int max_point_size, int max_cov_points_size, float planer_threshold, double min_plane_likeness)
    : max_layer_(max_layer), layer_(layer), layer_point_size_(layer_point_size), max_points_size_(max_point_size), max_cov_points_size_(max_cov_points_size), planer_threshold_(planer_threshold), min_plane_likeness_(min_plane_likeness) {
  temp_points_.clear();
  octo_state_     = 0;
  new_points_num_ = 0;
  all_points_num_ = 0;
  // when new points num > 5, do a update
  update_size_threshold_      = 5;
  init_octo_                  = false;
  update_enable_              = true;
  update_cov_enable_          = true;
  max_plane_update_threshold_ = layer_point_size_[layer_];
  for (int i = 0; i < 8; i++) {
    leaves_[i] = nullptr;
  }
  plane_ptr_ = new Plane;
}

// check is plane , calc plane parameters including plane covariance
void OctoTree::InitPlane(const std::vector<pointWithCovMeta> &points, Plane *plane) {
  plane->covariance  = Eigen::Matrix3d::Zero();
  plane->center      = Eigen::Vector3d::Zero();
  plane->normal      = Eigen::Vector3d::Zero();
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
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);
  Eigen::Matrix3d                                evecs    = es.eigenvectors().real();
  Eigen::Vector3d                                evals    = es.eigenvalues().real();
  Eigen::Matrix3f::Index                         evalsMin = 0, evalsMid = 1, evalsMax = 2;  // SelfAdjointEigenSolver's eigen values are in increase order
  // plane covariance calculation
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
      1.0 / plane->points_size;
  double plane_likeness = 2 * (evals(evalsMid) - evals(evalsMin)) / evals.sum();
  if (evals(evalsMin) < planer_threshold_ && plane_likeness > min_plane_likeness_) {
    plane->is_plane = true;
  } else {
    plane->is_plane = false;
  }

  plane->normal          = evecs.col(evalsMin);
  plane->y_normal        = evecs.col(evalsMid);
  plane->x_normal        = evecs.col(evalsMax);
  plane->min_eigen_value = evals(evalsMin);
  plane->mid_eigen_value = evals(evalsMid);
  plane->max_eigen_value = evals(evalsMax);
  plane->radius          = sqrt(evals(evalsMax));
  plane->d               = -plane->normal.dot(plane->center);

  plane->id = g_plane_id++;
}

// only updaye plane normal, center and radius with new points
void OctoTree::UpdatePlane(const std::vector<pointWithCovMeta> &points, Plane *plane) {
  Eigen::Matrix3d old_covariance = plane->covariance;
  Eigen::Vector3d old_center     = plane->center;
  Eigen::Matrix3d sum_ppt =
      (plane->covariance + plane->center * plane->center.transpose()) *
      plane->points_size;
  Eigen::Vector3d sum_p = plane->center * plane->points_size;
  for (size_t i = 0; i < points.size(); i++) {
    Eigen::Vector3d pv = points[i].pw;
    sum_ppt += pv * pv.transpose();
    sum_p += pv;
  }
  plane->points_size = plane->points_size + points.size();
  plane->center      = sum_p / plane->points_size;
  plane->covariance  = sum_ppt / plane->points_size -
                      plane->center * plane->center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);
  Eigen::Matrix3d                                evecs    = es.eigenvectors().real();
  Eigen::Vector3d                                evals    = es.eigenvalues().real();
  Eigen::Matrix3f::Index                         evalsMin = 0, evalsMid = 1, evalsMax = 2;

  plane->normal          = evecs.col(evalsMin);
  plane->y_normal        = evecs.col(evalsMid);
  plane->x_normal        = evecs.col(evalsMax);
  plane->min_eigen_value = evals(evalsMin);
  plane->mid_eigen_value = evals(evalsMid);
  plane->max_eigen_value = evals(evalsMax);
  plane->radius          = sqrt(evals(evalsMax));
  plane->d               = -plane->normal.dot(plane->center);

  if (evals(evalsMin) < planer_threshold_) {
    plane->is_plane = true;
  } else {
    plane->is_plane = false;
  }
}

void OctoTree::InitOctoTree() {
  if (temp_points_.size() > max_plane_update_threshold_) {
    InitPlane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane == true) {
      octo_state_ = 0;
      if (temp_points_.size() > max_cov_points_size_) {
        update_cov_enable_ = false;
      }
      if (temp_points_.size() > max_points_size_) {
        update_enable_ = false;
      }
      // note by kk: here we force to split voxel
      CutOctoTree();
    } else {
      octo_state_ = 1;
      CutOctoTree();
    }
    init_octo_      = true;
    new_points_num_ = 0;
    //      temp_points_.clear();
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
          max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
          max_cov_points_size_, planer_threshold_, min_plane_likeness_);
      leaves_[leafnum]->voxel_center_[0] =
          voxel_center_[0] + (2 * xyz[0] - 1) * quarter_length_;
      leaves_[leafnum]->voxel_center_[1] =
          voxel_center_[1] + (2 * xyz[1] - 1) * quarter_length_;
      leaves_[leafnum]->voxel_center_[2] =
          voxel_center_[2] + (2 * xyz[2] - 1) * quarter_length_;
      leaves_[leafnum]->quarter_length_ = quarter_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_num_++;
  }
  for (uint i = 0; i < 8; i++) {
    if (leaves_[i] != nullptr) {
      if (leaves_[i]->temp_points_.size() >
          leaves_[i]->max_plane_update_threshold_) {
        InitPlane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane) {
          leaves_[i]->octo_state_ = 0;
        } else {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->CutOctoTree();
        }
        leaves_[i]->init_octo_      = true;
        leaves_[i]->new_points_num_ = 0;
      }
    }
  }
}

void BuildVoxelMap(const std::vector<pointWithCovMeta> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float                               planer_threshold,
                   double                                    min_plane_likeness,
                   std::unordered_map<VoxelLoc, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCovMeta p_v = input_points[i];
    float                  loc_xyz[3];
    // 1. 计算点所在网格
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.pw[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                      (int64_t)loc_xyz[2]);
    auto     iter = feat_map.find(position);
    // 2. 把点加入网格八叉树中
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold, min_plane_likeness);
      feat_map[position]                   = octo_tree;
      feat_map[position]->quarter_length_  = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
  }
  // 3. 初始化八叉树
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->InitOctoTree();
  }
}

struct M_POINT {
  Eigen::Vector3d              center;
  int                          count = 0;
  std::vector<Eigen::Vector3d> points;
};

template <typename T>
void DownSamplingVoxel(const pcl::PointCloud<PointType> &cloud_in,
                       pcl::PointCloud<PointType>       &cloud_out,
                       double                            voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  std::unordered_map<VoxelLoc, M_POINT> feat_map;

  for (uint i = 0; i < cloud_in.size(); i++) {
    Eigen::Vector3d p_c = cloud_in[i].getVector3fMap().cast<double>();
    int             loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1;
      }
    }

    VoxelLoc position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
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

  std::unordered_map<VoxelLoc, M_POINT> feat_map;

  for (uint i = 0; i < cloud_in.size(); i++) {
    Eigen::Vector3d p_c = cloud_in[i].getVector3fMap().cast<double>();
    int             loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1;
      }
    }

    VoxelLoc position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
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

void OctoTree::ExtractSurfelInfo(std::vector<pcl::PointCloud<PointType>> &cloud_multi_layers, std::vector<Surfel> &surfels, int cur_layer) {
  if (this->plane_ptr_ && this->plane_ptr_->is_plane) {
    for (auto &e : this->temp_points_) {
      PointType p;
      p.getVector3fMap() = e.pw.cast<float>();
      p.intensity        = this->plane_ptr_->id;
      p.time             = this->plane_ptr_->min_eigen_value;
      cloud_multi_layers[cur_layer].push_back(p);
    }
    if (!this->temp_points_.empty()) {
      if (this->temp_points_.back().timestamp - this->temp_points_.front().timestamp < 0.07) {
        Surfel sf;
        sf.timestamp  = 0;  // todo kk
        sf.resolution = this->quarter_length_ * 4;
        sf.center     = this->plane_ptr_->center;
        sf.covariance = this->plane_ptr_->covariance;
        sf.norm       = this->plane_ptr_->normal;
        surfels.push_back(sf);
      } else {
        LOG(WARNING) << "Surfel dropped because of non-proximal timestamps: " << this->temp_points_.back().timestamp - this->temp_points_.front().timestamp;
      }
    }
  }
  for (auto &e : this->leaves_) {
    if (e) {
      e->ExtractSurfelInfo(cloud_multi_layers, surfels, cur_layer + 1);
    }
  }
}

void BuildSurfels(const pcl::PointCloud<hilti_ros::Point>::Ptr &cloud, std::vector<Surfel> &surfels) {
  std::unordered_map<VoxelLoc, OctoTree *> feat_map;
  std::vector<PointWithCovMeta>            points;

  for (auto &e : *cloud) {
    PointWithCovMeta np;
    np.timestamp = e.time;
    np.pw        = e.getVector3fMap().cast<double>();
    points.push_back(np);
  }

  BuildVoxelMap(points, 0.8, 3, {20, 20, 20, 20}, 1000, 1000, 0.005, 0.1, feat_map);

  std::vector<pcl::PointCloud<PointType>> cloud_surfel_multi_layers(4);
  for (auto &e : feat_map) {
    e.second->ExtractSurfelInfo(cloud_surfel_multi_layers, surfels);
    delete e.second;
  }

  // static int i = 0;
  // pcl::io::savePLYFileBinary(std::to_string(i) + "-0.ply", cloud_surfel_multi_layers[0]);
  // pcl::io::savePLYFileBinary(std::to_string(i) + "-1.ply", cloud_surfel_multi_layers[1]);
  // pcl::io::savePLYFileBinary(std::to_string(i) + "-2.ply", cloud_surfel_multi_layers[2]);
  // pcl::io::savePLYFileBinary(std::to_string(i) + "-3.ply", cloud_surfel_multi_layers[3]);
  // ++i;

  LOG(INFO) << "Surfel Extraction done: surfel count = " << surfels.size();
}
