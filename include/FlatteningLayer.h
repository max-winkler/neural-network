#ifndef _FLATTENING_LAYER_H_
#define _FLATTENING_LAYER_H_

#include "Layer.h"

class FlatteningLayer : public Layer
{
 public:
  FlatteningLayer(size_t, size_t);

  void eval(DataArray*&) const override;
  void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const override;
  std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>&,
					    const std::vector<DataArray*>&,
					    const std::vector<DataArray*>&) const override;
  
  std::unique_ptr<Layer> clone() const override;
  std::unique_ptr<Layer> zeros_like() const override;

  void save(std::ostream&) const override;
  
 private:
  size_t in_dim1;
  size_t in_dim2;
};

#endif
