#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <vector>
#include <random>
#include <memory>

#include "Layer.h"
#include "Matrix.h"
#include "Vector.h"
#include "Activation.h"
#include "TrainingData.h"

// Struct for optimization options used in Neuralnetwork::train
struct OptimizationOptions
{
  OptimizationOptions();
  
  size_t max_iter;
  size_t batch_size;
  float learning_rate;
  size_t output_every;
  size_t epochs;
  
  enum LossFunction {
    MSE, LOG
  } loss_function;
    
};

// Main class representing a neural network
class NeuralNetwork
{
 public:
  
  /**
   * Default constructor creating an empty (0 layers) neural network.
   */
  NeuralNetwork();

  /**
   * Constructor initializing a neural network with \p l layers.
   *
   * @param l The number of layers the neural network should have (including input layer)
   */
  NeuralNetwork(size_t l);

  /**
   * Copy constructor creating a hard copy of \p other.
   *
   * @param other The neural network to be copied.
   */
  NeuralNetwork(const NeuralNetwork& other);

  /**
   * Move constructor replacing the current instance by \p other.
   *
   * @param other The neural network to be moved.
   */
  NeuralNetwork(NeuralNetwork&& other);

  /**
   * Move assignment operator replacing the current instance by \p other.
   *
   * @param other The neural network to be moved.
   */
  NeuralNetwork& operator=(NeuralNetwork&& other);
  
  // Static creator function with zero-initialization but correct dimensions
  static NeuralNetwork createLike(const NeuralNetwork&);

  // Add neural network layers
  /**
   * Adds an input layer with input dimension \p i x \p j to the neural network.
   * If \p j is 0, a vector input layer is created, otherwise a matrix input layer.
   *
   * @param i The number of components in the first dimension
   * @param j The number of components in the second dimension (optional). 
   */
  void addInputLayer(size_t i, size_t j=0);

  /**
   * Adds a pooling layer to the neural network. The currently implemented method is the max pooling.
   *
   * @param batch The number of rows/columns of the batched for the pooling operation.
   */
  void addPoolingLayer(size_t batch, size_t S=0);

  /**
   * Adds a flattening layer to the neural network. Works only when the previous layer outputs matrix-valued data.
   */
  void addFlatteningLayer();

  /**
   * Adds a convolutional layer to the neural network. The output of the previous layer must be matrix-valued.
   *
   * @param F the number of feature maps.
   * @param batch The number of rows/columns of the kernel matrix to be used.
   * @param act The activation function applied component-wise after the convolution operation.
   * @param S The stride used in the convolution operation (default value is 1).
   * @param P The padding used in the convolution operation (default value is 0).
   */
  void addConvolutionLayer(size_t F, size_t batch, ActivationFunction act, size_t S=1, size_t P=0);

  /**
   * Adds a fully connected layer to the neural network. The both input and output are vector-valued.
   *
   * @param i The output dimension of the layer.
   * @param act The activation function.
   */
  void addFullyConnectedLayer(size_t i, ActivationFunction act);

  /**
   * Adds a fully connexted layer but with softmax activation function returning probabilities of how likely the 
   * input belongs to the respective class.
   * 
   * @param i The output dimension of the layer (equal to the number of classes).
   */
  void addClassificationLayer(size_t i);

  /**
   * Initializes the neural network after the arcitecture using addInputLayer(), addFullyConnectedLayer(), etc. 
   * has been called. The weights and biases are initialized with random values.
   */
  void initialize();

  /**
   * Returns the number of layers of the neural network.
   */
  size_t n_layers() const;
  
  /**
   * Computes the output of the neural network for the given input.
   *
   * @param input Input data to be evaluated. Must be a matrix when the input layer is of type \c MatrixInputLayer
   * and a vector when the input layer is of type \c VectorInputLayer.
   */
  Vector eval(const DataArray& input) const;

  /**
   * Train the neural network for the given training data. This method implements a mini batch stochastic gradient
   * using a forward propagation implemented in evalFunctional() and for the gradient computation a backward
   * propagation implemented in evalGradient().
   *
   * @param data A vector of training data to which the neural network will be matched.
   * @param options A collection of parameters used in the optimization routine. If omitted reasonable default
   * values will be used.
   */
  void train(const std::vector<TrainingData>& data, OptimizationOptions options=OptimizationOptions());

  /**
   * Evaluate the loss function used for the training of the neural network for a set of training data.
   * 
   * @param data The training data set used for the training.
   * @param y Auxiliary output variable that will be reused in evalGradient().
   * @param z Auxiliary output variable that will be reused in evalGradient().
   * @param batch_idx Indices of the training samples belonging to the batch for which we want to 
   * evaluate the neural network.
   * @param options Collection of parameters for fine tuning of the optimization algorithm.
   */
  float evalFunctional(const std::vector<TrainingData>& data,
		    std::vector<std::vector<DataArray*>>& y,
		    std::vector<std::vector<DataArray*>>& z,
		    const std::vector<size_t>& batch_idx,
		    OptimizationOptions options) const;
  
  /**
   * Evaluate the gradient of the loss function used for the training of the neural network.
   *
   * @param data The training data set used for the training.
   * @param y Auxiliary data produced in evalFunctional() that can be reused here.
   * @param z Auxiliary data produced in evalFunctional() that can be reused here.
   * @param batch_idx Indices of the training samples belonging to the batch for which we want to 
   * evaluate the neural network.
   * @param options Collection of parameters for fine tuning of the optimization algorithm.
   */
  NeuralNetwork evalGradient(const std::vector<TrainingData>& data,
			      const std::vector<std::vector<DataArray*>>& y,
			      const std::vector<std::vector<DataArray*>>& z,
			      const std::vector<size_t>& batch_idx,
			      OptimizationOptions options) const;

  /**
   * Returns the scalar product of two neural networks. This is basically the sum of the 
   * inner products of all weight and kernel matrices and bias vectors in the neural nerwork.
   *
   * @param other Right argument of the scalar product.
   */
  float dot(const NeuralNetwork& other) const;

  /**
   * Returns the norm of a neural network which is the square root of the scalar product 
   * of the neural network with itself.
   */
  float norm() const;  

  // Console output
  friend std::ostream& operator<<(std::ostream&, const NeuralNetwork&);

  /**
   * Stores the neural network arcitecture and all the weight and kernel matrices and
   * bias vectors in a file.
   */
  void save(const std::string&) const;
  void save(std::ostream&) const;
  
 private:
  // Layer list of neural network
  std::vector<std::unique_ptr<Layer>> layers;

  // Initialization state of the neural network
  bool initialized;
  
  // Update step with gradient method
  void update_increment(float momentum, const NeuralNetwork& gradient, float learning_rate);
  void apply_increment(const NeuralNetwork& increment);
  
  // Gradient test (for debugging and testing)
  void gradientTest(const NeuralNetwork&,
		    const std::vector<TrainingData>&,
		    const std::vector<size_t>&,
		    OptimizationOptions) const;
};

#endif
