#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(size_t in_dim1, size_t in_dim2,
			   size_t k, size_t S, size_t P)
  : Layer(std::vector<size_t>(2, 0), LayerType::POOLING),
    in_dim1(in_dim1), in_dim2(in_dim2),
    k(k), S(S==0 ? k : S), P(P), type(POOLING_MAX)
{
  // Apply simple pooling to get the output dimension of the layer
  Matrix A(in_dim1, in_dim2);
  Matrix B = A.pool(type, S, P);

  dim[0] = B.nRows();
  dim[1] = B.nCols();
}

void PoolingLayer::eval(DataArray*& x_) const
{
  Matrix& x = dynamic_cast<Matrix&>(*x_);
  x = x.pool(type, S, P);
}

void PoolingLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Matrix& x = dynamic_cast<const Matrix&>(x_);
  Matrix& y = dynamic_cast<Matrix&>(y_);

  y = x.pool(type, S, P);
}


std::unique_ptr<Layer> PoolingLayer::backward_propagate(std::vector<DataArray*>& DY,
							const std::vector<DataArray*>& Y,
							const std::vector<DataArray*>& Z) const
{
  PoolingLayer* output = new PoolingLayer(in_dim1, in_dim2, k, S, P);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Matrix& Dy = dynamic_cast<Matrix&>(**Dy_it);
      const Matrix& y = dynamic_cast<const Matrix&>(**y_it);
      const Matrix& z = dynamic_cast<const Matrix&>(**z_it);

      Dy = Dy.unpool(y, type, S, P);
    }

  return std::unique_ptr<Layer>(output);
}

std::unique_ptr<Layer> PoolingLayer::clone() const
{
  return std::unique_ptr<Layer>(new PoolingLayer(*this));
}

std::unique_ptr<Layer> PoolingLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new PoolingLayer(*this));
}
