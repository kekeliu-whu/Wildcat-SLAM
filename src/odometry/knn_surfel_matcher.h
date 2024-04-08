#pragma once

#include <gtest/gtest.h>
#include <deque>
#include <flann/flann.hpp>
#include <vector>

#include "surfel.h"

class KnnSurfelMatcher {
  FRIEND_TEST(KnnSurfelMatcher, KNearestSearch);

 public:
  using FloatType  = double;
  using FLANNIndex = flann::Index<flann::L2_Simple<FloatType>>;

  void BuildIndex(const std::deque<Surfel::Ptr> &surfels);

  void Match(std::deque<Surfel::Ptr> &surfels, std::vector<SurfelCorrespondence> &surfels_corrs);

  void KNearestSearch(const Surfel::Ptr &surfel, int k, std::vector<Surfel::Ptr> &k_nearest_surfels);

 private:
  void FLANNBuildIndex(const std::vector<FloatType> &cloud);

  void FLANNKNearestSearch(std::vector<FloatType> &query, int k, std::vector<int> &k_indices, std::vector<FloatType> &k_distances);

  std::vector<FloatType> ToVector(const Surfel::Ptr &surfel);

 private:
  std::deque<Surfel::Ptr> target_surfels_;

  std::vector<FloatType>      cloud_;
  std::shared_ptr<FLANNIndex> index_;
  int                         dim_ = 6;

  static constexpr double kSpatialKnnNormalize        = 1.0;
  static constexpr double kAngularKnnNormalize        = 5.0 * M_PI / 180.0;
  static constexpr double kNormDistThreshold          = 0.2;
  static constexpr double kProjDistThreshold          = 0.5;
  static constexpr int    kNearestSurfelCandidatesNum = 10;
  static constexpr double kTimeDiffThreshold          = 0.06;
};
