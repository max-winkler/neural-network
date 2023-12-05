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
  double learning_rate;
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
  
  // Constructors
  NeuralNetwork();
  NeuralNetwork(size_t);
  NeuralNetwork(const NeuralNetwork&);
  NeuralNetwork(NeuralNetwork&&);

  /// Assignment operators
  NeuralNetwork& operator=(NeuralNetwork&&);
  
  // Static creator function with zero-initialization but correct dimensions
  static NeuralNetwork createLike(const NeuralNetwork&);

  // Add neural network layers
  void addInputLayer(size_t i, size_t j=0);
  void addPoolingLayer(size_t);
  void addFlatteningLayer();
  void addConvolutionLayer(size_t, ActivationFunction, size_t S=1, size_t P=0);
  void addFullyConnectedLayer(size_t, ActivationFunction);
  void addClassificationLayer(size_t);

  // Initialization of neural network, set weights randomly
  void initialize();
  size_t n_layers() const;
  
  // Evaluate neural network in given point
  Vector eval(const DataArray&) const;

  // Train neural network for according to given training data
  void train(const std::vector<TrainingData>&, OptimizationOptions options=OptimizationOptions());

  // Functional evaluation (for training routine)
  double evalFunctional(const std::vector<TrainingData>&,
			 std::vector<std::vector<DataArray*>>&,
			 std::vector<std::vector<DataArray*>>&,
			 const std::vector<size_t>&,
			 OptimizationOptions) const;
  
  // Gradient evaluation (for training routine)
  NeuralNetwork evalGradient(const std::vector<TrainingData>&,
			      const std::vector<std::vector<DataArray*>>&,
			      const std::vector<std::vector<DataArray*>>&,
			      const std::vector<size_t>&,
			      OptimizationOptions) const;

  // Scalar product and norm (used in gradient test and stopping criterion)
  double dot(const NeuralNetwork&) const;
  double norm() const;  

  // Console output
  friend std::ostream& operator<<(std::ostream&, const NeuralNetwork&);

  // File IO
  void save(const std::string&) const;
  
 private:
  // Layer list of neural network
  std::vector<std::unique_ptr<Layer>> layers;

  // Initialization state of the neural network
  bool initialized;
  
  // Update step with gradient method
  void update_increment(double momentum, const NeuralNetwork& gradient, double learning_rate);
  void apply_increment(const NeuralNetwork& increment);
  
  // Gradient test (for debugging and testing)
  void gradientTest(const NeuralNetwork&,
		    const std::vector<TrainingData>&,
		    const std::vector<size_t>&,
		    OptimizationOptions) const;
};

#endif
