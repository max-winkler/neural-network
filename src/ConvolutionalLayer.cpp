#include <iomanip>

#include "ConvolutionalLayer.h"

#include "LinAlg.h"
#include "Random.h"

ConvolutionalLayer::ConvolutionalLayer(std::vector<size_t> in_dim, size_t F,
			         size_t k, size_t S, size_t P,
			         ActivationFunction act)
  : Layer(std::vector<size_t>(3,0), LayerType::CONVOLUTION),
    in_dim(in_dim), k(k), S(S==0 ? k : S), P(P),
    K(F), bias(F), act(act)
{
  // Set up kernel tensors
  for(size_t i=0; i<F; ++i)
    K[i] = Tensor(in_dim[0], k, k);    
  
  // Apply a simple convolution to get the output dimension of the layer
  Matrix A(in_dim[1], in_dim[2]);
  Matrix B = linalg::convolve(A, Matrix(k,k), S, P);

  dim[0] = F;
  dim[1] = B.nRows();
  dim[2] = B.nCols();
}

void ConvolutionalLayer::eval(DataArray*& x_) const
{
  Tensor& x = dynamic_cast<Tensor&>(*x_);
  Tensor y(dim[0], dim[1], dim[2]);

  // Apply convolution and add bias
  for(size_t c=0; c<dim[0]; ++c)
    {
      y[c] = linalg::tensor_convolve(x, K[c], S, P);
      y[c] += bias[c];      
    }

  // Apply activation function
  // TODO: Think about more efficient implementation and shorter notation
  for(size_t c=0; c<y.nChannels(); ++c)
    for(size_t i=0; i<y.nRows(); ++i)
      for(size_t j=0; j<y.nCols(); ++j)
        y(c,i,j) = activate(y(c,i,j), act);

  delete x_;
  x_ = new Tensor(std::move(y));
}

void ConvolutionalLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Tensor& x = dynamic_cast<const Tensor&>(x_);
  Tensor& z = dynamic_cast<Tensor&>(z_);
  Tensor& y = dynamic_cast<Tensor&>(y_);
  
  for(size_t c=0; c<y.nChannels(); ++c)
    {
      z[c] = linalg::tensor_convolve(x, K[c], S, P);
      z[c] += bias[c];
    }
  
  for(size_t c=0; c<y.nChannels(); ++c)
    for(size_t i=0; i<y.nRows(); ++i)
      for(size_t j=0; j<y.nCols(); ++j)
        y(c,i,j) = activate(z(c,i,j), act);
}

std::unique_ptr<Layer> ConvolutionalLayer::backward_propagate(std::vector<DataArray*>& DY,
						  const std::vector<DataArray*>& Y,
						  const std::vector<DataArray*>& Z) const
{
  ConvolutionalLayer* output = new ConvolutionalLayer(in_dim, K.size(), k, S, P, act);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(; y_it != Y.end(); ++y_it, ++z_it, ++Dy_it)
    {
      Tensor& Dy = dynamic_cast<Tensor&>(**Dy_it);            
      
      const Tensor& y = dynamic_cast<const Tensor&>(**y_it);
      const Tensor& z = dynamic_cast<const Tensor&>(**z_it);

      Tensor Dx(y.nChannels(), y.nRows(), y.nCols());
      
      Tensor Dz(z);
      for(size_t c=0; c<Dz.nChannels(); ++c)
        for(size_t i=0; i<Dz.nRows(); ++i)
	for(size_t j=0; j<Dz.nCols(); ++j)
	  Dz(c,i,j) = Dactivate(Dz(c,i,j), act);
      
      const int F = K.size();
      
      for(size_t d=0; d<F; ++d)
        {
	MatrixView Dz_slice = Dz[d];
	MatrixView Dy_slice = Dy[d];
	  
	output->K[d] += linalg::tensor_convolve(y, linalg::multiply(Dz_slice, Dy_slice), S, P);
	output->bias[d] += linalg::dot(Dz_slice, Dy_slice);

	for(size_t c=0; c<y.nChannels(); ++c)
	  Dx[c] += linalg::convolve(linalg::multiply(Dz_slice, Dy_slice), K[d][c], 1, k-1, true);
        }

      delete *Dy_it;
      *Dy_it = new Tensor(std::move(Dx));
    }
  
  return  std::unique_ptr<Layer>(output);
}

float ConvolutionalLayer::dot(const Layer& other) const
{
  const ConvolutionalLayer& o = dynamic_cast<const ConvolutionalLayer&>(other);

  float s = 0;
  for(auto e = K.begin(), f=o.K.begin();
      e != K.end(); ++e, ++f)
    s += (*e).inner(*f);
  for(auto b = bias.begin(), c = o.bias.begin(); b != bias.end(); ++b, ++c)
    s += (*b)*(*c);

  return s;
}

void ConvolutionalLayer::initialize()
{
  float a = sqrt(4.0f / (in_dim[0] * k * k));
  
  Random gen = Random::create_normal_random_generator(0.0f, a);
  for(auto e = K.begin(); e != K.end(); ++e)
    for(size_t c=0; c<(*e).nChannels(); ++c)
      for(size_t i=0; i<(*e).nRows(); ++i)
        for(size_t j=0; j<(*e).nCols(); ++j)
	(*e)(c,i,j) = gen();

  gen = Random::create_uniform_random_generator();
  for(auto b = bias.begin(); b != bias.end(); ++b)
    (*b) = 0.3*gen();
}

void ConvolutionalLayer::update_increment(float momentum, const Layer& grad_layer_, float learning_rate)
{
  const ConvolutionalLayer& grad_layer = dynamic_cast<const ConvolutionalLayer&>(grad_layer_);

  for(size_t k=0; k<K.size(); ++k)
    {
      K[k] *= momentum;
      K[k] += learning_rate*grad_layer.K[k];
    }

  size_t i=0;
  for(auto& b: bias)
    {      
      b *= momentum;
      b += learning_rate*grad_layer.bias[i++];
    }
}

void ConvolutionalLayer::apply_increment(const Layer& inc_layer_)
{
  const ConvolutionalLayer& inc_layer = dynamic_cast<const ConvolutionalLayer&>(inc_layer_);

  for(size_t k=0; k<K.size(); ++k)
    {
      K[k] -= inc_layer.K[k];
      bias[k] -= inc_layer.bias[k];
    }
}

std::unique_ptr<Layer> ConvolutionalLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new ConvolutionalLayer(in_dim, K.size(), k, S, P, act));
}

std::unique_ptr<Layer> ConvolutionalLayer::clone() const
{
  return std::unique_ptr<Layer>(new ConvolutionalLayer(*this));
}

void ConvolutionalLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << " dimension : " << dim[0] << ", " << dim[1] << ", " << dim[2] << '\n';

  os << std::setw(16) << " kernel : ";
  for(auto e = K.begin(); e != K.end(); ++e)
    {
      const Tensor& T = *e;
      for(size_t d=0; d<T.nChannels(); ++d)
        for(size_t i=0; i<T.nRows(); ++i)
	for(size_t j=0; j<T.nCols(); ++j)
	  os << T(d,i,j) << ", ";
    
      os << '\n';
    }

  os << std::setw(16) << " bias : ";
  for(auto b = bias.begin(); b != bias.end(); ++b)
    os << *b << ", ";
  os << '\n';
  
  os << std::setw(16) << " activation : " << act << '\n';  

  os << std::setw(16) << " stride : " << S << '\n';
  os << std::setw(16) << " padding : " << P << '\n';
}

std::map<std::string, std::string> ConvolutionalLayer::get_parameters() const
{
  return {
    {"activation", ActivationFunctionName.at(act)},
    {"kernelsize",  std::to_string(k)},
    {"stride",     std::to_string(S)},
    {"padding",    std::to_string(P)},
    {"features",   std::to_string(K.size())}
  };
}

std::map<std::string, std::pair<const float*, std::vector<size_t>>> ConvolutionalLayer::get_weights() const
{
  std::map<std::string, std::pair<const float*, std::vector<size_t>>> weights;

  size_t c = K[0].nChannels();
  size_t m = K[0].nRows();
  size_t n = K[0].nCols();
  
  for(size_t i=0; i<K.size(); ++i)
    {
      std::ostringstream ss;
      ss << "kernel_" << std::setw(2) << std::setfill('0') << i;

      weights[ss.str()] = std::make_pair(&K[i](0,0,0), std::vector<size_t>{c,m,n});
    }

  weights["bias"]   =  std::make_pair(bias.data(), std::vector<size_t>{bias.size()});

  return weights;
}

void ConvolutionalLayer::set_weights(const std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>& weights)
{
  size_t F = K.size();
  
  // Set bias vector
  bias = weights.at("bias").first;

  for(size_t i=0; i<F; ++i)
    {
      // Retrieve kernel weight details
      std::stringstream ss;
      ss << "kernel_" << std::setw(2) << std::setfill('0') << i;

      const std::pair<std::vector<float>, std::vector<size_t>>&
        weight_details = weights.at(ss.str().c_str());
      
      // Set kernels
      const std::vector<float>& weight_data = weight_details.first;
      const std::vector<size_t>& weight_dim = weight_details.second;

      K[i] = Tensor(weight_dim[0], weight_dim[1], weight_dim[2], weight_data.data());
    }
}
