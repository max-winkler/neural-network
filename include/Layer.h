#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <unordered_map>

#include "Activation.h"
#include "Matrix.h"

// Forward declarations for friend definitions
class NeuralNetwork;
class ScaledNeuralNetwork;

enum LayerType
  {
    VECTOR_INPUT,
    MATRIX_INPUT,
    FULLY_CONNECTED,
    CLASSIFICATION,
    CONVOLUTION,
    POOLING,
    FLATTENING
  };

class Layer
{
 public:
  Layer(std::pair<size_t, size_t>, LayerType, ActivationFunction);
  
  double dot(const Layer&) const;

  static std::unordered_map<LayerType, const char*> LayerName;
 private:
  std::pair<size_t, size_t> dimension;

  // Matrix and vector (might be unused depending on layer type)
  Matrix weight;
  Vector bias;

  // Stright and padding (might be unused depending on layer type)
  size_t m;
  size_t S;
  size_t P;
  
  ActivationFunction activation_function;

  LayerType layer_type;

  friend class NeuralNetwork;
  friend class ScaledNeuralNetwork;  
  friend std::ostream& operator<<(std::ostream&, const Layer&);
  friend NeuralNetwork operator+(const NeuralNetwork& lhs, const NeuralNetwork& rhs);
  friend NeuralNetwork operator+(const NeuralNetwork& lhs, const ScaledNeuralNetwork& rhs);
};

#endif
