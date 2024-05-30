#include <iomanip>

#include "ConvolutionalLayer.h"
#include "Random.h"

ConvolutionalLayer::ConvolutionalLayer(size_t in_dim1, size_t in_dim2,
				       size_t k, size_t S, size_t P, ActivationFunction act)
  : Layer(std::vector<size_t>(2,0), LayerType::CONVOLUTION),
    in_dim1(in_dim1), in_dim2(in_dim2),
    k(k), S(S==0 ? k : S), P(P), K(k,k), bias(0.), act(act)
{
  // Apply a simple convolution to get the dimension of the layer
  Matrix A(in_dim1, in_dim2);
  Matrix B = A.convolve(K, S, P);

  dim[0] = B.nRows();
  dim[1] = B.nCols();
}

void ConvolutionalLayer::eval(DataArray*& x_) const
{
  Matrix& x = dynamic_cast<Matrix&>(*x_);
  x = x.convolve(K, S, P);
  x += bias;
  x = activate(x, act);  
}

void ConvolutionalLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Matrix& x = dynamic_cast<const Matrix&>(x_);
  Matrix& z = dynamic_cast<Matrix&>(z_);
  Matrix& y = dynamic_cast<Matrix&>(y_);
  
  z = x.convolve(K, S, P);
  z += bias;
  y = activate(z, act);
}

std::unique_ptr<Layer> ConvolutionalLayer::backward_propagate(std::vector<DataArray*>& DY,
							      const std::vector<DataArray*>& Y,
							      const std::vector<DataArray*>& Z) const
{
  ConvolutionalLayer* output = new ConvolutionalLayer(in_dim1, in_dim2, k, S, P, act);
  

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Matrix& Dy = dynamic_cast<Matrix&>(**Dy_it);
      const Matrix& y = dynamic_cast<const Matrix&>(**y_it);
      const Matrix& z = dynamic_cast<const Matrix&>(**z_it);
      
      Matrix Dz = Dactivate(z, act);
      
      // Gradient w.r.t. kernel matrix and bias
      output->K += y.back_convolve(multiply(Dz, Dy), S, P);
      output->bias += Dy.inner(Dz);
      
      // Gradient w.r.t. data
      Dy = multiply(Dz, Dy).kron(K);
    }

  return  std::unique_ptr<Layer>(output);
}

double ConvolutionalLayer::dot(const Layer& other) const
{
  const ConvolutionalLayer& b = dynamic_cast<const ConvolutionalLayer&>(other);
  return K.inner(b.K) + bias*b.bias;
}

void ConvolutionalLayer::initialize()
{
  double a = 0.6;
  
  Random gen = Random::create_uniform_random_generator();
  for(size_t i=0; i<K.nRows(); ++i)
    for(size_t j=0; j<K.nCols(); ++j)
      K[i][j] = -a+2.*a*gen();
}

void ConvolutionalLayer::update_increment(double momentum, const Layer& grad_layer_, double learning_rate)
{
  const ConvolutionalLayer& grad_layer = dynamic_cast<const ConvolutionalLayer&>(grad_layer_);
  
  K *= momentum;
  K += learning_rate*grad_layer.K;
  
  bias *= momentum;
  bias += learning_rate*grad_layer.bias;
}

void ConvolutionalLayer::apply_increment(const Layer& inc_layer_)
{
  const ConvolutionalLayer& inc_layer = dynamic_cast<const ConvolutionalLayer&>(inc_layer_);

  // TODO: Implement operator-= instead
  K += (-1.)*inc_layer.K;
  bias += (-1.)*inc_layer.bias;
}

std::unique_ptr<Layer> ConvolutionalLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new ConvolutionalLayer(in_dim1, in_dim2, k, S, P, act));
}

std::unique_ptr<Layer> ConvolutionalLayer::clone() const
{
  return std::unique_ptr<Layer>(new ConvolutionalLayer(*this));
}

void ConvolutionalLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << " dimension : " << dim[0] << ", " << dim[1] << '\n';

  os << std::setw(16) << " kernel : ";
  for(size_t i=0; i<K.nRows(); ++i)
    for(size_t j=0; j<K.nCols(); ++j)
      os << K[i][j] << ", ";
  os << '\n';

  os << std::setw(16) << " bias : " << bias << '\n';  
  os << std::setw(16) << " activation : " << act << '\n';  

  os << std::setw(16) << " stride : " << S << '\n';
  os << std::setw(16) << " padding : " << P << '\n';
}
