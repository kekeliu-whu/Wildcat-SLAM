#include "feature_map.h"

std::shared_ptr<FeatureMap> FeatureMap::Create(
    const std::deque<Surfel::Ptr>     &surfels,
    int                                layers,
    double                             resolution,
    std::vector<SurfelCorrespondence> &surfel_correspondences) {
  auto map = std::make_shared<FeatureMap>(layers, resolution);
  for (auto &surfel : surfels) {
    int      layer = map->GetLayerByResolution(surfel->resolution);
    VoxelLoc loc{surfel->GetCenterInWorld(), surfel->resolution};
    if (map->cloud_surfel_multi_layers_[layer].find(loc) == map->cloud_surfel_multi_layers_[layer].end()) {
      map->cloud_surfel_multi_layers_[layer][loc] = surfel;
    } else {
      SurfelCorrespondence corr{map->cloud_surfel_multi_layers_[layer][loc], surfel};
      surfel_correspondences.push_back(corr);
      if (surfel->plane_std_deviation < map->cloud_surfel_multi_layers_[layer][loc]->plane_std_deviation) {
        map->cloud_surfel_multi_layers_[layer][loc] = surfel;
      }
    }
  }
  return map;
}
