#ifndef _VECTOR_INPUT_LAYER_H_
#define _VECTOR_INPUT_LAYER_H_

#include "Layer.h"

class VectorInputLayer : public Layer
{
 public:
  VectorInputLayer(size_t);

  DataArray eval(const DataArray&) const override;
  void eval_functional(const DataArray& x, DataArray& z, DataArray& y) const override;
  Layer backpropagate(std::vector<DataArray*>&,
		      const std::vector<DataArray*>&,
		      const std::vector<DataArray*>&) const override;
  
 private:
};

#endif