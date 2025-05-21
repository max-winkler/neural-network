#ifndef _FULLY_CONNECTED_LAYER_H_
#define _FULLY_CONNECTED_LAYER_H_

#include "Layer.h"

class FullyConnectedLayer : public Layer
{  
  friend NeuralNetwork;
public:
  FullyConnectedLayer(size_t, size_t, ActivationFunction);
  
  void eval(DataArray*&) const override;

  void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const override;
  
  std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>&,
					    const std::vector<DataArray*>&,
					    const std::vector<DataArray*>&) const override;  
  
  
  float dot(const Layer&) const override;
  
  void initialize() override;  
  
  void update_increment(float, const Layer&, float) override;
  void apply_increment(const Layer&) override;
  
  std::unique_ptr<Layer> clone() const override;
  std::unique_ptr<Layer> zeros_like() const override;

  void save(std::ostream&) const override;

  std::map<std::string, std::string> get_parameters() const override;
  std::map<std::string, std::pair<const float*, std::vector<size_t>>> get_weights() const override;

  static std::unique_ptr<Layer> create_from_parameters(const std::vector<size_t>&,
                                                       const std::vector<size_t>&,
                                                       const std::map<std::string, std::string>&,
                                                       const std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>&);
private:
  
  // Layer-specific parameters
  Matrix weight;
  Vector bias;
  ActivationFunction act;  
};

#endif

