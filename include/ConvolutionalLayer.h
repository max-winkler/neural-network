#ifndef _CONVOLUTIONAL_LAYER_H_
#define _CONVOLUTIONAL_LAYER_H_

#include "Layer.h"

#include "Tensor.h"
#include "Matrix.h"

/**
 * Class for convolution layers handling matrix-value input and computing the convolution with a kernel
 * matrix. The kernel matrix is the parameter to be trained.
 */
class ConvolutionalLayer : public Layer
{
 public:
  /**
   * Constructor creating a convolutional layer. Given input parameters are the dimensions of the input data,
   * the size of the kernel matrix, the stride and padding parameter and the activation function to be used.
   *
   * @brief Create convolutional layer instance
   *
   * @param in_dim Shape of the input tensor.
   * @param F Number of features.
   * @param k Width/height of the kernel matrix.
   * @param S Stride parameter.
   * @param P Padding parameter.
   */
  ConvolutionalLayer(std::vector<size_t>, size_t, size_t,
		 size_t S=0, size_t P=0,
		 ActivationFunction act=ActivationFunction::NONE);

  /**
   * Compute the output of the layer for provided input data.
   *
   * @brief Evaluate the layer
   *
   * @param x_ Input and output data of the layer (will be overwritten).
   */
  void eval(DataArray*&) const override;

  /**
   * Perform a forward propagation for this layer.
   *
   * @brief Forward propagation for the layer.
   *
   * @param x_ The input data array. Must be of type Matrix.
   * @param z_ Output data before the activation function is applied.
   * @param y_ Output data of the layer.
   */
  void forward_propagate(const DataArray& x, DataArray& z, DataArray& y) const override;

  /**
   * Perform a backward propagation for this layer.
   *
   * @brief Backward propagation for the layer.
   *
   * @param DY Set of data to be processed. The output is further processed by the previous layer.
   * @param Y Set of input data. Each entry must be of type Matrix.
   * @param Z The z values computed before by forward_propagate(const DataArray&, DataArray&, DataArray&).
   */
  std::unique_ptr<Layer> backward_propagate(std::vector<DataArray*>&,
					    const std::vector<DataArray*>&,
					    const std::vector<DataArray*>&) const override;  
  
  /**
   * Computes the dot product with another layer. 
   *
   * @brief Compute to product
   *
   * @param other The second argument of the dot product.
   */
  float dot(const Layer&) const override;

  /**
   * Initialize the layer and fill the parameters with random data.
   *
   * @brief Initialize layer
   */
  void initialize() override;  

  /**
   * Update the increment for the optimization routine.
   */
  void update_increment(float, const Layer&, float) override;

  /**
   * Apply the increment \p inc_layer_ to the current one.
   */
  void apply_increment(const Layer&) override;

  /**
   * Create a copy of the layer.
   */
  std::unique_ptr<Layer> clone() const override;

  /**
   * Create another layer with the same dimension and zero parameters.
   */
  std::unique_ptr<Layer> zeros_like() const override;  

  /**
   * Save the layer to text written in the output stream.
   */
  void save(std::ostream&) const override;

  std::map<std::string, std::string> get_parameters() const override;
  std::map<std::string, std::pair<const float*, std::vector<size_t>>> get_weights() const override;

  void set_weights(const std::map<std::string, std::pair<std::vector<float>, std::vector<size_t>>>&) override;
  
private:
  
  std::vector<Tensor> K;
  std::vector<float> bias;
  ActivationFunction act;
  
  size_t k;
  size_t S;
  size_t P;

  std::vector<size_t> in_dim;
};

#endif
