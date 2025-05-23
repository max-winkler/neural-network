#include <iomanip>

#include "MatrixInputLayer.h"
#include "Tensor.h"

MatrixInputLayer::MatrixInputLayer(size_t m, size_t n)
  : Layer(std::vector<size_t>({1, m, n}), LayerType::MATRIX_INPUT)
{
}

void MatrixInputLayer::eval(DataArray*& x) const
{
}

void MatrixInputLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Tensor& x = dynamic_cast<const Tensor&>(x_);
  Tensor& y = dynamic_cast<Tensor&>(y_);
  
  y = x;  
}

std::unique_ptr<Layer> MatrixInputLayer::backward_propagate(std::vector<DataArray*>& DY,
						const std::vector<DataArray*>& Y,
						const std::vector<DataArray*>& Z) const
{
  return std::unique_ptr<Layer>(new MatrixInputLayer(dim[1], dim[2]));
}

std::unique_ptr<Layer> MatrixInputLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new MatrixInputLayer(dim[1], dim[2]));
}

std::unique_ptr<Layer> MatrixInputLayer::clone() const
{
  return std::unique_ptr<Layer>(new MatrixInputLayer(*this));
}

void MatrixInputLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << "  dimension : " << dim[1] << ", " << dim[2] << '\n';  
}

