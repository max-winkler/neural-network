#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstring>

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
    FLATTENING,
    UNKNOWN
  };

class Layer
{
 public:
  
  // Process layers
  virtual DataArray eval(const DataArray&) const;
  virtual void eval_functional(const DataArray& x, DataArray& z, DataArray& y) const;
			       
  // Get gradient
  virtual Layer backpropagate(std::vector<DataArray*>&,
			      const std::vector<DataArray*>&,
			      const std::vector<DataArray*>&) const;
  
  virtual double dot(const Layer&) const;

  virtual void initialize();
  
  virtual void update(double, const Layer&, double);
  
  static std::unordered_map<LayerType, const char*> LayerName;
  std::string get_name() const;

  virtual Layer operator+(const Layer&);
protected:
  
  Layer(std::vector<size_t>, LayerType);
  
  std::vector<size_t> dim;
  LayerType layer_type;
  
private:

  // LayerType layer_type; // Do we need this when using inheritence?

  friend class NeuralNetwork;
  friend class ScaledNeuralNetwork;  
  friend std::ostream& operator<<(std::ostream&, const Layer&);
  friend NeuralNetwork operator+(const NeuralNetwork& lhs, const NeuralNetwork& rhs);
  friend NeuralNetwork operator+(const NeuralNetwork& lhs, const ScaledNeuralNetwork& rhs);
};

#endif
