#include "Layer.h"

std::unordered_map<LayerType, const char*> Layer::LayerName =
  {
    {VECTOR_INPUT, "Vector Input Layer"},
    {MATRIX_INPUT, "Matrix Input Layer"},
    {FULLY_CONNECTED, "Fully connected Layer"},
    {CLASSIFICATION, "Classification Layer"},
    {CONVOLUTION, "Convolution Layer"},
    {POOLING, "Pooling Layer"},
    {FLATTENING, "Flattening Layer"}
  };

Layer::Layer(std::pair<size_t, size_t> dimension, LayerType layer_type, ActivationFunction activation_function)
  : dimension(dimension), layer_type(layer_type), activation_function(activation_function),
    weight(Matrix()), bias(Vector())
{}

double Layer::dot(const Layer& rhs) const
{
  double val = 0.;

  switch(layer_type)
    {
    case FULLY_CONNECTED:
    case CLASSIFICATION:
      
      val += weight.inner(rhs.weight);
      val += bias.inner(rhs.bias);		
                
      break;

    case VECTOR_INPUT:
    case MATRIX_INPUT:
      // Input layers have no weight and bias
      break;
    default:
      std::cerr << "ERROR: Inner product for layer type " << LayerName[layer_type] << " not implemented yet.\n";
    }
  return val;
}

std::ostream& operator<<(std::ostream& os, const Layer& layer)
{
  os << Layer::LayerName[layer.layer_type] << " (";
  os << layer.dimension.first;
  // Print second dimension if present
  switch(layer.layer_type)
    {
    case MATRIX_INPUT:
    case CONVOLUTION:
    case POOLING:
      os << " x " << layer.dimension.second ;
    }
  os << " neurons)\n";

  return os;
}
