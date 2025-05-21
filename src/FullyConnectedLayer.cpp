#include <iomanip>

#include "FullyConnectedLayer.h"
#include "Random.h"

FullyConnectedLayer::FullyConnectedLayer(size_t dim, size_t in_dim, ActivationFunction act)
  : Layer(std::vector<size_t>(1, dim), LayerType::FULLY_CONNECTED),
    act(act), bias(dim), weight(dim, in_dim)
{
}

void FullyConnectedLayer::eval(DataArray*& x_) const
{
  Vector* x = dynamic_cast<Vector*>(x_);

  *x = activate(weight * (*x) + bias, act);
}

void FullyConnectedLayer::forward_propagate(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Vector& x = dynamic_cast<const Vector&>(x_);
  Vector& z = dynamic_cast<Vector&>(z_);
  Vector& y = dynamic_cast<Vector&>(y_);

  z = weight * x + bias;
  y = activate(z, act);
}

std::unique_ptr<Layer> FullyConnectedLayer::backward_propagate(std::vector<DataArray*>& DY,
						   const std::vector<DataArray*>& Y,
						   const std::vector<DataArray*>& Z) const
{
  FullyConnectedLayer* output = new FullyConnectedLayer(dim[0], weight.nCols(), act);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Vector& Dy = dynamic_cast<Vector&>(**Dy_it);
      const Vector& y = dynamic_cast<const Vector&>(**y_it);
      const Vector& z = dynamic_cast<const Vector&>(**z_it);

      Vector Dz;

      switch(act)
	{
	case ActivationFunction::SOFTMAX:
	  // Activation functions taking all components into account
	  Dz = Dy * DactivateCoupled(z, act);
	  break;
	  
	default:
	  // Activation functions applied component-wise
	  Vector tmp(Dactivate(z, act));
	  Dz = Dy * diag(tmp);
	  break;
	}
      
      // Update gradient w.r.t. weight and bias
      output->weight += outer(Dz, y);
      output->bias += Dz;

      // Update gradient w.r.t. input data
      Dy = Dz * weight;
    }

  return std::unique_ptr<Layer>(output);
}

float FullyConnectedLayer::dot(const Layer& other) const
{
  const FullyConnectedLayer& b = dynamic_cast<const FullyConnectedLayer&>(other);
  
  return weight.inner(b.weight) + bias.inner(b.bias);
}

void FullyConnectedLayer::initialize()
{
  float a = 2.;
  
  Random gen = Random::create_uniform_random_generator();
  for(size_t i=0; i<weight.nRows(); ++i)
    for(size_t j=0; j<weight.nCols(); ++j)
      weight(i,j) = -a+2.*a*gen();
  
  // std::cout << "Initial weight: \n" << weight << std::endl;
}

void FullyConnectedLayer::update_increment(float momentum, const Layer& grad_layer_, float learning_rate)
{
  const FullyConnectedLayer& grad_layer = dynamic_cast<const FullyConnectedLayer&>(grad_layer_);
  
  weight *= momentum;
  weight += learning_rate*grad_layer.weight;
  
  bias *= momentum;
  bias += learning_rate*grad_layer.bias;
}

void FullyConnectedLayer::apply_increment(const Layer& inc_layer_)
{
  const FullyConnectedLayer& inc_layer = dynamic_cast<const FullyConnectedLayer&>(inc_layer_);

  // TODO: Implement operator-= instead
  weight += (-1.)*inc_layer.weight;
  bias += (-1.)*inc_layer.bias;
}

std::unique_ptr<Layer> FullyConnectedLayer::zeros_like() const
{
  return std::unique_ptr<Layer>(new FullyConnectedLayer(dim[0], weight.nCols(), act));
}

std::unique_ptr<Layer> FullyConnectedLayer::clone() const
{
  return std::unique_ptr<Layer>(new FullyConnectedLayer(*this));
}

void FullyConnectedLayer::save(std::ostream& os) const
{
  os << "[ " << get_name() << " ]\n";
  os << std::setw(16) << " dimension : " << dim[0] << '\n';

  os << std::setw(16) << " weight : ";
  for(size_t i=0; i<weight.nRows(); ++i)
    for(size_t j=0; j<weight.nCols(); ++j)
      os << weight[i][j] << ", ";
  os << '\n';
  
  os << std::setw(16) << " bias : ";
  for(size_t i=0; i<bias.length(); ++i)
    os << bias[i] << ", ";
  os << '\n';

  os << std::setw(16) << " activation : " << act << '\n';  
}

std::map<std::string, std::string> FullyConnectedLayer::get_parameters() const
{
  return {
    {"activation", ActivationFunctionName.at(act)}
  };
}

std::map<std::string, std::pair<const float*, std::vector<size_t>>> FullyConnectedLayer::get_weights() const
{
  std::map<std::string, std::pair<const float*, std::vector<size_t>>> weights;
  weights["weight"] =  std::make_pair(&weight[0][0], std::vector<size_t>{weight.nRows(), weight.nCols()});
  weights["bias"]   =  std::make_pair(&bias[0],      std::vector<size_t>{bias.nEntries()});

  return weights;
}

std::unique_ptr<Layer> FullyConnectedLayer::create_from_parameters(const std::vector<size_t>& dim,
                                                                   const std::vector<size_t>& in_dim,
                                                                   const std::map<std::string, std::string>& parameters,
                                                                   const std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>& weights)
{
  // Create layer
  auto* layer = new FullyConnectedLayer(dim[0], in_dim[0], ActivationFunctionFromName.at(parameters.at("activation")));

  // Check dimensions
  if(dim[0] != weights.at("weight").second[0] || in_dim[0] != weights.at("weight").second[1])
    {
      std::cerr << "ERROR: Inconsistent matrix dimension in fully connected layer.\n";
      std::exit(EXIT_FAILURE);      
    }

  if(dim[0] != weights.at("bias").second[0])
    {
      std::cerr << "ERROR: Inconsistent vector dimension in fully connected layer.\n";
      std::exit(EXIT_FAILURE);
    }

  // Set weights
  layer->weight = Matrix(dim[0], in_dim[0], weights.at("weight").first.data());
  layer->bias = Vector(dim[0], weights.at("bias").first.data());

  return std::unique_ptr<Layer>(layer);
}
