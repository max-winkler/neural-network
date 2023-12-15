#include "MatrixInputLayer.h"

MatrixInputLayer::MatrixInputLayer(size_t dim1, size_t dim2)
  : Layer(std::vector({dim1, dim2}), LayerType::MATRIX_INPUT)
{
}

void MatrixInputLayer::forward_propagate(DataArray*& x) const
{
}

void MatrixInputLayer::eval_functional(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Matrix& x = dynamic_cast<const Matrix&>(x_);
  Matrix& y = dynamic_cast<Matrix&>(y_);

  y = x;  
}

std::unique_ptr<Layer> MatrixInputLayer::backpropagate(std::vector<DataArray*>& DY,
						       const std::vector<DataArray*>& Y,
						       const std::vector<DataArray*>& Z) const
{
  MatrixInputLayer output(dim[0], dim[1]);
  return std::unique_ptr<Layer>(new MatrixInputLayer(dim[0], dim[1]));
}

std::unique_ptr<Layer> MatrixInputLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new MatrixInputLayer(dim[0], dim[1]));
}

std::unique_ptr<Layer> MatrixInputLayer::clone() const
{
  return std::unique_ptr<Layer>(new MatrixInputLayer(*this));
}

