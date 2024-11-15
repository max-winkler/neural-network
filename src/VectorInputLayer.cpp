#include <iomanip>

#include "VectorInputLayer.h"

VectorInputLayer::VectorInputLayer(size_t dim)
  : Layer(std::vector<size_t>(1, dim), LayerType::VECTOR_INPUT)
{  
}

void VectorInputLayer::eval(DataArray*& x) const
{
}

void VectorInputLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Vector& x = dynamic_cast<const Vector&>(x_);
  Vector& y = dynamic_cast<Vector&>(y_);
  
  y = x;
}

std::unique_ptr<Layer> VectorInputLayer::backward_propagate(std::vector<DataArray*>& DY,
							    const std::vector<DataArray*>& Y,
							    const std::vector<DataArray*>& Z) const
{
  VectorInputLayer output(dim[0]);
  return std::unique_ptr<Layer>(new VectorInputLayer(dim[0]));
}

std::unique_ptr<Layer> VectorInputLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new VectorInputLayer(dim[0]));
}

std::unique_ptr<Layer> VectorInputLayer::clone() const
{
  return std::unique_ptr<Layer>(new VectorInputLayer(*this));
}

void VectorInputLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << "  dimension : " << dim[0] << '\n';  
}
