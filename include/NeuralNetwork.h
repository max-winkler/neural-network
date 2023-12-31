#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <vector>
#include <random>

#include "Layer.h"
#include "Matrix.h"
#include "Vector.h"
#include "Activation.h"
#include "TrainingData.h"

// Forward class declarations
struct ScaledNeuralNetwork;

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
  NeuralNetwork(NeuralNetwork&&);

  /// Assignment operators
  NeuralNetwork& operator=(NeuralNetwork&&);
  NeuralNetwork& operator=(const ScaledNeuralNetwork&);
  
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

  // Operations on neural network parameters
  NeuralNetwork& operator*=(double);
  NeuralNetwork& operator+=(const ScaledNeuralNetwork&);
  friend ScaledNeuralNetwork operator*(double, const NeuralNetwork&);      
  friend NeuralNetwork operator+(const NeuralNetwork&, const NeuralNetwork&);  
  friend NeuralNetwork operator+(const NeuralNetwork& lhs, const ScaledNeuralNetwork& rhs);
    
  // Console output
  friend std::ostream& operator<<(std::ostream&, const NeuralNetwork&);

  // File IO
  void save(const std::string&) const;
  
 private:
  // Layer list of neural network
  std::vector<Layer> layers;  

  // Initialization state of the neural network
  bool initialized;
  
  // Random number generator
  std::mt19937 rnd_gen;
  std::uniform_real_distribution<> random_real;
  std::normal_distribution<> random_normal;
  
  // Gradient test (for debugging and testing)
  void gradientTest(const NeuralNetwork&,
		    const std::vector<TrainingData>&,
		    const std::vector<size_t>&,
		    OptimizationOptions) const;
};

struct ScaledNeuralNetwork
{
  ScaledNeuralNetwork(double, const NeuralNetwork&);
  double scale;
  const NeuralNetwork* network;
};

#endif
