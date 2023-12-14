#ifndef _FLATTENING_LAYER_H_
#define _FLATTENING_LAYER_H_

class FlatteningLayer : public Layer
{
 public:
  FlatteningLayer(size_t);

  void forward_propagate(DataArray&) const override;
  void eval_functional(const DataArray& x, DataArray& z, DataArray& y) const override;
  std::unique_ptr<Layer> backpropagate(std::vector<DataArray*>&,
				       const std::vector<DataArray*>&,
				       const std::vector<DataArray*>&) const override;
  
  std::unique_ptr<Layer> clone() const override;
  std::unique_ptr<Layer> zeros_like() const override;

 private:
};

#endif
