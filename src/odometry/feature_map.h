#pragma once

#include <absl/container/flat_hash_map.h>

#include "odometry/surfel_extraction.h"
#include "surfel.h"

class FeatureMap {
 public:
  FeatureMap(int layers, double resolution) : layers_(layers), resolutions_(resolution) {
    cloud_surfel_multi_layers_.resize(layers);
    for (int i = 0; i < layers; ++i) {
      resolutions_.push_back(resolution / std::pow(2, i));
    }
  }

  int GetLayerByResolution(double r) {
    return log2(resolutions_[0] / r + 1e-6);
  }

 public:
  void AddFeature(Surfel::Ptr &surfel) {
  }

  void SearchFeaturePairs() {
  }

  static std::shared_ptr<FeatureMap> Create(
      const std::deque<Surfel::Ptr>     &surfels,
      int                                layers,
      double                             resolution,
      std::vector<SurfelCorrespondence> &surfel_correspondences);

 private:
  int                 layers_;
  std::vector<double> resolutions_;

  std::vector<absl::flat_hash_map<VoxelLoc, Surfel::Ptr>> cloud_surfel_multi_layers_;
};
