#include "knn_surfel_matcher.h"

void KnnSurfelMatcher::BuildIndex(const std::deque<Surfel::Ptr> &surfels) {
  if (surfels.empty()) {
    return;
  }
  target_surfels_ = surfels;
  std::vector<FloatType> cloud;
  for (auto &surfel : target_surfels_) {
    std::vector<FloatType> vec = ToVector(surfel);
    cloud.insert(cloud.end(), vec.begin(), vec.end());
  }
  this->FLANNBuildIndex(cloud);
}

void KnnSurfelMatcher::Match(std::deque<Surfel::Ptr> &surfels, std::vector<SurfelCorrespondence> &surfels_corrs) {
  surfels_corrs.clear();
  if (target_surfels_.empty()) {
    return;
  }
  std::set<std::pair<Surfel::Ptr, Surfel::Ptr>> surfel_pairs;
  for (auto &surfel : surfels) {
    std::vector<Surfel::Ptr> k_nearest_surfels;
    this->KNearestSearch(surfel, kNearestSurfelCandidatesNum, k_nearest_surfels);
    for (auto &nearest_surfel : k_nearest_surfels) {
      if (std::abs(nearest_surfel->timestamp - surfel->timestamp) < kTimeDiffThreshold) {
        continue;
      }
      if (surfel->AngularDistance(*nearest_surfel) > kAngularDistThreshold) {
        continue;
      }
      if (std::abs(surfel->normal.dot(surfel->center - nearest_surfel->center)) > kSurfelDistThreshold) {
        continue;
      }
      if (surfel_pairs.find({surfel, nearest_surfel}) != surfel_pairs.end() ||
          surfel_pairs.find({nearest_surfel, surfel}) != surfel_pairs.end()) {
        continue;
      }
      surfel_pairs.insert({surfel, nearest_surfel});

      if (surfel->timestamp < nearest_surfel->timestamp) {
        surfels_corrs.push_back({surfel, nearest_surfel});
      } else {
        surfels_corrs.push_back({nearest_surfel, surfel});
      }
      break;
    }
  }
}

void KnnSurfelMatcher::KNearestSearch(const Surfel::Ptr &surfel, int k, std::vector<Surfel::Ptr> &k_nearest_surfels) {
  std::vector<FloatType> query = ToVector(surfel);
  CHECK_EQ(query.size(), dim_);

  std::vector<int>       k_indices;
  std::vector<FloatType> k_distances;

  this->FLANNKNearestSearch(query, k, k_indices, k_distances);

  for (int i = 0; i < k; ++i) {
    k_nearest_surfels.push_back(target_surfels_[k_indices[i]]);
  }
}

void KnnSurfelMatcher::FLANNBuildIndex(const std::vector<FloatType> &cloud) {
  CHECK_EQ(cloud.size() % dim_, 0);
  cloud_ = cloud;  // save cloud_ for flann search
  index_.reset(
      new FLANNIndex(
          flann::Matrix<FloatType>(cloud_.data(), cloud_.size() / dim_, dim_),
          flann::KDTreeSingleIndexParams(15)));
  index_->buildIndex();
}

void KnnSurfelMatcher::FLANNKNearestSearch(std::vector<FloatType> &query, int k, std::vector<int> &k_indices, std::vector<FloatType> &k_distances) {
  CHECK(query.size() == dim_);

  k_indices.resize(k);
  k_distances.resize(k);

  flann::Matrix<int>       k_indices_mat(&k_indices[0], 1, k);
  flann::Matrix<FloatType> k_distances_mat(&k_distances[0], 1, k);

  // Wrap the k_indices and k_distances vectors (no data copy)
  index_->knnSearch(
      flann::Matrix<FloatType>(&query[0], 1, dim_),
      k_indices_mat, k_distances_mat, k,
      flann::SearchParams(-1, 0.0));
}

std::vector<KnnSurfelMatcher::FloatType> KnnSurfelMatcher::ToVector(const Surfel::Ptr &surfel) {
  // todo use resolution
  auto     center         = surfel->center;
  auto     norm           = surfel->normal;
  Vector3d center_uniform = center / kCenterDistThreshold;
  Vector3d norm_uniform   = norm / kAngularDistThreshold;
  return {center_uniform.x(), center_uniform.y(), center_uniform.z(), norm_uniform.x(), norm_uniform.y(), norm_uniform.z()};
}
