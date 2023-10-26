#include "knn_surfel_matcher.h"

namespace {

using FloatType = KnnSurfelMatcher::FloatType;
using Vector6   = Eigen::Matrix<KnnSurfelMatcher::FloatType, 6, 1>;

std::vector<FloatType> GetRandomVector() {
  Vector6                ret = Vector6::Random();
  std::vector<FloatType> vec;
  for (int i = 0; i < 6; ++i) {
    vec.push_back(ret[i]);
  }
  return vec;
}

}  // namespace

TEST(KnnSurfelMatcher, KNearestSearch) {
  KnnSurfelMatcher sm;

  std::vector<std::vector<FloatType>> vecs;
  for (int i = 0; i < 10'000; ++i) {
    vecs.push_back(GetRandomVector());
  }

  std::vector<FloatType> cloud;
  for (auto &v : vecs) {
    for (int i = 0; i < 6; ++i) {
      cloud.push_back(v[i]);
    }
  }

  sm.FLANNBuildIndex(cloud);

  for (int i = 0; i < vecs.size(); ++i) {
    std::vector<int>       k_indices;
    std::vector<FloatType> k_distances;
    sm.FLANNKNearestSearch(vecs[i], 10, k_indices, k_distances);
    EXPECT_EQ(k_indices.size(), 10);
    EXPECT_EQ(k_indices[0], i);
  }
}
