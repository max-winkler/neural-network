#include "FlatteningLayer.h"

FlatteningLayer::FlatteningLayer(size_t dim)
  : Layer(std::vector<size_t>(1, dim), Layer::FLATTENING_LAYER)
{
}

void FlatteningLayer::forward_propagate(DataArray& x_) const
{
  x_ = dynamic_cast<Matrix&>(x_).flatten();
}

void FlatteningLayer::eval_functional(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Matrix& x = dynamic_cast<const Matrix&>(x_);
  Vector& y = dynamic_cast<Vector&>(y_);

  y = x.flatten();
}

std::unique_ptr<Layer> FlatteningLayer::backpropagate(std::vector<DataArray*>& DY,
						      const std::vector<DataArray*>& Y,
						      const std::vector<DataArray*>& Z) const
{
  FlatteningLayer* output = new FlatteningLayer(dim[0]);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Matrix& Dy = dynamic_cast<Matrix&>(**Dy_it);
      const Vector& y = dynamic_cast<const Vector&>(**y_it);
      const Vector& z = dynamic_cast<const Vector&>(**z_it);

      // Dy = Matrix(Dy.reshape(
      // How to reshape? I do not know the original dimension and have no access to the previous layer. I must store
      // These values somewhere.
    }
}
