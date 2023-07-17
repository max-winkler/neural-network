#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<size_t> width_) : width(width_),
							   weight(), bias(), activation(),
							   layers(width_.size())
{
  // Add output dimension
  width.push_back(1);

  weight.reserve(layers);
  bias.reserve(layers);
  activation.reserve(layers);
  
  for(Dimension::const_iterator it = width.begin(); it+1 != width.end(); ++it)
    {
      weight.push_back(Matrix(*(it+1), *it));
      bias.push_back(Vector(*(it+1)));
    }
}

void NeuralNetwork::setParameters(size_t layer, const Matrix& matrix, const Vector& vector, ActivationFunction act)
{
  if(layer < 0 || layer > layers-1)
    {
      std::cerr << "Error: The layer " << layer << " is not feasible. Must be between 0 and " << layers-1 << std::endl;
      return;
    }

  if(matrix.nRows() != width[layer+1] || matrix.nCols() != width[layer])
    {
      std::cerr << "Error: The provided weight matrix has invalid size.\n";
      std::cerr << "  Given is (" << matrix.nRows() << "," << matrix.nCols() << ")"
		<< " but required is (" << width[layer+1] << "," << width[layer] << ")\n";
      return;
    }
  
  if(vector.size() != width[layer+1])
    {
      std::cerr << "Error: The provided bias vector has invalid size.\n";
      std::cerr << "  Given is (" << vector.size() 
		<< " but required is (" << width[layer+1] << "," << ")\n";
      return;
    }
  
  weight[layer] = matrix;
  bias[layer] = vector;
  activation[layer] = act;
}

double NeuralNetwork::eval(const Vector& x) const
{
  Vector x_tmp(x);
  for(size_t l=0; l<layers; ++l)    
    x_tmp = activate(weight[l] * x_tmp + bias[l], activation[l]);

  return x_tmp[0];
}

void NeuralNetwork::train(const std::vector<TrainingData>& data)
{
  // TODO: Implement training algorithm
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net) 
{
  os << "Neural network with (" << (net.layers) << " layers)\n";
  os << "  Input dimension         : " << (net.width[0]) << "\n";

  os << "  Hidden layer dimensions : ";
  for(size_t i=1; i<net.layers-1; ++i)    
    os << net.width[i] << ", ";
  os << net.width[net.layers-1] << std::endl;
  
  return os;
}
