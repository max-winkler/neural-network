#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <memory>

#include "Activation.h"
#include "Matrix.h"

// Forward declarations for friend definitions
class NeuralNetwork;

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
  virtual void forward_propagate(DataArray&) const;
  virtual void eval_functional(const DataArray& x, DataArray& z, DataArray& y) const;
			       
  // Get gradient
  virtual std::unique_ptr<Layer> backpropagate(std::vector<DataArray*>&,
					       const std::vector<DataArray*>&,
					       const std::vector<DataArray*>&) const;
  
  virtual double dot(const Layer&) const;
  
  virtual void initialize();
  virtual void update_increment(double, const Layer&, double);
  virtual void apply_increment(const Layer&);
  virtual std::unique_ptr<Layer> zeros_like() const;
  
  static std::unordered_map<LayerType, const char*> LayerName;
  std::string get_name() const;
  
protected:
  
  Layer(std::vector<size_t>, LayerType);
  
  std::vector<size_t> dim;
  LayerType layer_type;
  
private:

  friend NeuralNetwork;
  friend std::ostream& operator<<(std::ostream&, const Layer&);
};

#endif
