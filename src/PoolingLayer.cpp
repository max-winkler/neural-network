#include <iomanip>
#include <cstdlib>

#include "PoolingLayer.h"

#include "LinAlg.h"

PoolingLayer::PoolingLayer(std::vector<size_t> in_dim,
		       size_t k, size_t S, size_t P)
  : Layer(std::vector<size_t>(3, 0), LayerType::POOLING),
    in_dim(in_dim), k(k), S(S==0 ? k : S), P(P), type(POOLING_MAX)
{
  S = this->S;
  
  // Check validity of parameters
  if((in_dim[1] - k) % S != 0 || (in_dim[2] - k) % S != 0)
  {
    std::cerr << "ERROR: Invalid parameters for pooling layer. Please choose kernel size and \n"
              << "       stride such that (n_in - k)/S is an integer. You choosed \n"
              << "       n_in = " << in_dim[1] << ", m_in = " << in_dim[2]
              << ", k = " << k << ", S = " << S << ".\n";
    std::exit(EXIT_FAILURE);
  }

  // Compute output dimension
  dim[0] = in_dim[0];
  dim[1] = (in_dim[1]-k)/S + 1;
  dim[2] = (in_dim[2]-k)/S + 1;
}

void PoolingLayer::eval(DataArray*& x_) const
{
  Tensor& x = dynamic_cast<Tensor&>(*x_);
  Tensor y(dim[0], dim[1], dim[2]);
  
  size_t d = x.nChannels();

  for(size_t c=0; c<d; ++c)
    {
      // TODO: Unnecessary copy operation here. The rvalue is copied. This can be more efficient.
      y[c] = linalg::pool(x[c], k, type, S, P);
    }

  delete x_;
  x_ = new Tensor(std::move(y));
}

void PoolingLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Tensor& x = dynamic_cast<const Tensor&>(x_);
  Tensor& y = dynamic_cast<Tensor&>(y_);

  size_t d = x.nChannels();
  
  for(size_t c=0; c<d; ++c)
    y[c] = linalg::pool(x[c], k, type, S, P);
}


std::unique_ptr<Layer> PoolingLayer::backward_propagate(std::vector<DataArray*>& DY,
					      const std::vector<DataArray*>& Y,
					      const std::vector<DataArray*>& Z) const
{
  PoolingLayer* output = new PoolingLayer(in_dim, k, S, P);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Tensor& Dy = dynamic_cast<Tensor&>(**Dy_it);      
      const Tensor& y = dynamic_cast<const Tensor&>(**y_it);

      Tensor Dx(y.nChannels(), y.nRows(), y.nCols());

      size_t d = Dy.nChannels();

      for(size_t c=0; c<d; ++c)
        Dx[c] = linalg::unpool(Dy[c], y[c], k, type, S, P);

      delete *Dy_it;
      *Dy_it = new Tensor(std::move(Dx));
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

void PoolingLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << " dimension : " << dim[0] << ", " << dim[1] << ", " << dim[2] << '\n';

  os << std::setw(16) << " batch size : " << k << '\n';
  os << std::setw(16) << " stride : " << S << '\n';
  os << std::setw(16) << " padding : " << P << '\n';
  os << std::setw(16) << " pooling type : " << type << '\n';
}

std::map<std::string, std::string> PoolingLayer::get_parameters() const
{
  return {
    {"stride",  std::to_string(S)},
    {"padding", std::to_string(P)},
    {"kernelsize", std::to_string(k)}
  };
}
