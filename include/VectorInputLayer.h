#ifndef _VECTOR_INPUT_LAYER_H_
#define _VECTOR_INPUT_LAYER_H_

#include "Layer.h"

class VectorInputLayer : public Layer
{
 public:
  VectorInputLayer(size_t);

  virtual DataArray eval(const DataArray&) const;
  
 private:
};

#endif
