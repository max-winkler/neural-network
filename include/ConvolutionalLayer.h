#ifndef _CONVOLUTIONAL_LAYER_H_
#define _CONVOLUTIONAL_LAYER_H_

#include "Layer.h"

class ConvolutionalLayer : public Layer
{
 public:  
  ConvolutionalLayer(size_t, size_t, size_t k, size_t S=0, size_t P=0, ActivationFunction act=ActivationFunction::NONE);
  
  void eval(DataArray*&) const override;

  void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const override;
  
  std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>&,
					    const std::vector<DataArray*>&,
					    const std::vector<DataArray*>&) const override;  
  
  
  double dot(const Layer&) const override;
  
  void initialize() override;  
  
  void update_increment(double, const Layer&, double) override;
  void apply_increment(const Layer&) override;
  
  std::unique_ptr<Layer> clone() const override;
  std::unique_ptr<Layer> zeros_like() const override;  

  void save(std::ostream&) const override;
 private:
  
  Matrix K;
  double bias;
  ActivationFunction act;
  
  size_t k;
  size_t S;
  size_t P;

  size_t in_dim1;
  size_t in_dim2;
};

#endif
