#ifndef _LAYER_H_
#define _LAYER_H_

#include <iostream>
#include <vector>
#include <map>
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
  virtual void eval(DataArray*&) const = 0;
  virtual void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const = 0;
			       
  // Get gradient
  virtual std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>&,
					  const std::vector<DataArray*>&,
					  const std::vector<DataArray*>&) const = 0;
  
  virtual float dot(const Layer&) const;
  
  virtual void initialize();
  virtual void update_increment(float, const Layer&, float);
  virtual void apply_increment(const Layer&);
  virtual std::unique_ptr<Layer> zeros_like() const = 0;
  virtual std::unique_ptr<Layer> clone() const = 0;
  
  static const std::unordered_map<LayerType, std::string> LayerName;
  static const std::unordered_map<LayerType, std::string> LayerShortName;
  static const std::unordered_map<std::string, LayerType> LayerTypeFromShortName;
  
  std::string get_name() const;

  /// deprecated
  virtual void save(std::ostream&) const; 

  virtual std::map<std::string, std::string> get_parameters() const;
  virtual std::map<std::string, std::pair<const float*, std::vector<size_t>>> get_weights() const;
  virtual void set_weights(const std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>&);
  
protected:
  
  Layer(std::vector<size_t>, LayerType);
  
  std::vector<size_t> dim;
  LayerType layer_type;
  
private:

  friend NeuralNetwork;
  friend std::ostream& operator<<(std::ostream&, const Layer&);
};

#endif
