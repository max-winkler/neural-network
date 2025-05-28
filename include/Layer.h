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

  /**
   * Evaluate the layer and overwrite the input.
   *
   * @brief Evaluate the layer.
   *
   * @param x Input/Output array of the layer.
   */
  virtual void eval(DataArray*&) const = 0;

  /**
   * Apply a forward propagation step for the current layer and store auxiliary data, required in the 
   * backward propagation routine, into arrays \p z and \p y.
   *
   * @brief Forward propagation.
   *
   * @param x Input data array for the layer.
   * @param z Output data before application of the activation function. Might be unused for some layers.
   * @param y Output data of the layer.
   */
  virtual void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const = 0;
			       
  /**
   * Apply backward-propagation step for a batch of data. Returns a pointer to a layer storing the
   * gradients with respect to all weights of the layer.
   *
   * @brief Backward propagation.
   *
   * @param DY Gradients of the proceeding layer with respect to their input (=output of current layer).
   *           Will be overwritten by the gradients w.r.t. the input of the current layer.
   * @param Y Auxiliary values y obtained in \p Layer::forward_propagate().
   * @param Z Auxiliary values y obtained in \p Layer::forward_propagate().
   */
  virtual std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>& DY,
                                                    const std::vector<DataArray*>& Y,
                                                    const std::vector<DataArray*>& Z) const = 0;

  /**
   * Compute the scalar product with another layer of the same type. In most cases this is the sum of the
   * scalar products of all weights of the two layers. The scalar product is used in optimization algorithms.
   *
   * @brief Scalar product of two layers.
   *
   * @param other The layer that is multiplied with the current instance.
   */
  virtual float dot(const Layer& other) const;

  /**
   * Initialize the layer. This assigns random values to the weights.
   *
   * @brief Initialize weights in a layer.
   */
  virtual void initialize();

  /**
   * In case the current instance is the increment used in a stochastic gradient method,
   * this routine computes the new increment as [\p momentum x (*\p this)+ \p learning_rate * \p grad_layer].
   * This routine usually forwards this operation to all the weights involved in the layer.
   * 
   * @brief Update the increment used in the momentum gradient method.
   *
   * @param momentum The momentum parameter choosed by the optimization routine.
   * @param grad_layer The layer storing the gradients w.r.t. the weights of the current layer.
   * Obtained by Layer::backward_propagate().
   * @param learning_rate The learning rate choosed by the optimization routine.
   */
  virtual void update_increment(float momentum, const Layer& grad_layer, float learning_rate);

  /**
   * Apply the increment according to [(*\p this) -= \p inc_layer]. This is used for the update step of the
   * momentum gradient method.
   *
   * @brief Apply the increment to the current layer.
   *
   * @param inc_layer The layer whose weights are subtracted from the weights of the current instance.
   */
  virtual void apply_increment(const Layer& inc_layer);

  /**
   * Create a layer of the same type but all weights initialized with zero.
   *
   * @brief Copy layer and initialize with zeros.
   */
  virtual std::unique_ptr<Layer> zeros_like() const = 0;

  /**
   * Create a copy of the current layer.
   *
   * @brief Create hard copy of the layer.
   */
  virtual std::unique_ptr<Layer> clone() const = 0;

  
  /// Map from LayerType to string representation of the layer.
  static const std::unordered_map<LayerType, std::string> LayerName; 
  /// Map from LayerType to short string representation of the layer. Used e.g. in XML input/output.   
  static const std::unordered_map<LayerType, std::string> LayerShortName;
  /// Map from short string representation of a layer to LayerType (inverse of Layer::LayerShortName).
  static const std::unordered_map<std::string, LayerType> LayerTypeFromShortName;

  /// Returns the layer type (as string) of the current layer.
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
