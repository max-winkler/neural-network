#ifndef _FULLY_CONNECTED_LAYER_H_
#define _FULLY_CONNECTED_LAYER_H_

#include "Layer.h"

class FullyConnectedLayer : public Layer
{  
 public:
  FullyConnectedLayer(size_t, size_t, ActivationFunction);
  
  DataArray eval(const DataArray&) const override;

  void eval_functional(const DataArray& x, DataArray& z, DataArray& y) const override;

  Layer backpropagate(std::vector<DataArray*>&,
		      const std::vector<DataArray*>&,
		      const std::vector<DataArray*>&) const override;  
  
  
  double dot(const Layer&) const override;
  
  void initialize() override;  
  
  void update_increment(double, const Layer&, double) override;
  void apply_increment(const Layer&) override;
  
 private:
  
  // Layer-specific parameters
  Matrix weight;
  Vector bias;
  ActivationFunction act;  
};

#endif

