#include "VectorInputLayer.h"

VectorInputLayer::VectorInputLayer(size_t dim)
  : Layer(std::vector(1, dim), LayerType::VECTOR_INPUT)
{  
}

DataArray VectorInputLayer::eval(const DataArray& input) const
{
  return input;
}

void VectorInputLayer::eval_functional(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Vector& x = dynamic_cast<const Vector&>(x_);
  Vector& y = dynamic_cast<Vector&>(y_);

  y = x;
}

Layer VectorInputLayer::backpropagate(std::vector<DataArray*>& DY,
				      const std::vector<DataArray*>& Y,
				      const std::vector<DataArray*>& Z) const
{
  VectorInputLayer output(dim[0]);
  return output;
}
