#ifndef _FLATTENING_LAYER_H_
#define _FLATTENING_LAYER_H_

#include "Layer.h"

class FlatteningLayer : public Layer
{
 public:
  FlatteningLayer(size_t, size_t);

  void forward_propagate(DataArray*&) const override;
  void eval_functional(const DataArray& x, DataArray& z, DataArray& y) const override;
  std::unique_ptr<Layer> backpropagate(std::vector<DataArray*>&,
				       const std::vector<DataArray*>&,
				       const std::vector<DataArray*>&) const override;
  
  std::unique_ptr<Layer> clone() const override;
  std::unique_ptr<Layer> zeros_like() const override;

 private:
  int in_dim1;
  int in_dim2;
};

#endif
