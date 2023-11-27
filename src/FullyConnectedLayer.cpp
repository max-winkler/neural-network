#include "FullyConnectedLayer.h"
#include "Random.h"

FullyConnectedLayer::FullyConnectedLayer(size_t dim, size_t in_dim, ActivationFunction act)
  : Layer(std::vector(1, dim), LayerType::FULLY_CONNECTED), act(act), bias(dim), weight(dim, in_dim)
{
}

DataArray FullyConnectedLayer::eval(const DataArray& input) const
{
  const Vector& x = dynamic_cast<const Vector&>(input);
  
  return activate(weight * x + bias, act);
}

void FullyConnectedLayer::eval_functional(const DataArray& x_, DataArray& z_, DataArray& y_) const
{
  const Vector& x = dynamic_cast<const Vector&>(x_);
  Vector& z = dynamic_cast<Vector&>(z_);
  Vector& y = dynamic_cast<Vector&>(y_);

  z = weight * x + bias;
  y = activate(z, act);
}

Layer FullyConnectedLayer::backpropagate(std::vector<DataArray*>& DY,
					 const std::vector<DataArray*>& Y,
					 const std::vector<DataArray*>& Z) const
{
  FullyConnectedLayer output(dim[0], weight.nCols(), act);

  auto y_it = Y.begin(), z_it = Z.begin();
  auto Dy_it = DY.begin();
  
  for(;  y_it != Y.end();
      ++y_it, ++z_it, ++Dy_it)
    {
      Vector& Dy = dynamic_cast<Vector&>(**Dy_it);
      const Vector& y = dynamic_cast<const Vector&>(**y_it);
      const Vector& z = dynamic_cast<const Vector&>(**z_it);
      
      Vector Dz = Dy * diag(Dactivate(z, act)); 

      // Update gradient w.r.t. weight and bias
      output.weight += outer(Dz, y);
      output.bias += Dz;

      // Update gradient w.r.t. input data
      Dy = Dz * weight;
    }

  return output;
}

double FullyConnectedLayer::dot(const Layer& other) const
{
  const FullyConnectedLayer& b = dynamic_cast<const FullyConnectedLayer&>(other);
  
  return weight.inner(b.weight) + bias.inner(b.weight);
}

void FullyConnectedLayer::initialize()
{
  for(auto it = weight.begin(); it != weight.end(); ++it)
    *it = -1.+2*Random::get_uniform();
}

void FullyConnectedLayer::update(double momentum, const Layer& grad_layer_, double learning_rate)
{
  const FullyConnectedLayer& grad_layer = dynamic_cast<const FullyConnectedLayer&>(grad_layer_);
  
  weight *= momentum;
  weight += (-learning_rate)*grad_layer.weight;
  
  bias *= momentum;
  bias += (-learning_rate)*grad_layer.bias;

  return res;  
}
