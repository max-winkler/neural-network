#ifndef _LAYER_FACTORY_H_
#define _LAYER_FACTORY_H_

#include "Layer.h"

using ParameterMap = std::map<std::string, std::string>;
using WeightMap = std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>;

class LayerFactory
{
 public: 
  /**
   * Create a layer of a given type from a parameter and weight map.
   *
   * @param type The type of the layer.
   * @param dim The output dimension of the layer.
   * @param in_dim The input dimension of the layer (output dimension of previous layer).
   * @param parameters Parameter map (depends on layer type).   
   * @param weight Weight map (depends on the layer type).
   */
  static std::unique_ptr<Layer> create(LayerType,
			         const std::vector<size_t>&,
			         const std::vector<size_t>&,
			         const ParameterMap&,
			         const WeightMap&);
};

#endif
