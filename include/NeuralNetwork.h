#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <vector>
#include <random>

#include "Layer.h"
#include "Matrix.h"
#include "Vector.h"
#include "Activation.h"
#include "TrainingData.h"

class NeuralNetwork;

struct ScaledNeuralNetwork
{
  ScaledNeuralNetwork(double, const NeuralNetwork&);
  double scale;
  const NeuralNetwork* network;

  //friend NeuralNetwork operator+(const ScaledNeuralNetwork&, const ScaledNeuralNetwork&);
};

struct OptimizationOptions
{
  OptimizationOptions();
  
  size_t max_iter;
  size_t batch_size;
  double learning_rate;
  
  enum LossFunction {
    MSE, LOG
  } loss_function;
    
};

class NeuralNetwork
{
 public:
  NeuralNetwork();
  NeuralNetwork(NeuralNetwork&&);
  
  NeuralNetwork& operator=(NeuralNetwork&&);
  NeuralNetwork& operator=(const ScaledNeuralNetwork&);
  
  // unused constructor: remove later
  //NeuralNetwork(Dimension);

  // static creator function
  static NeuralNetwork createLike(const NeuralNetwork&);
  
  void addInputLayer(size_t i, size_t j=0);
  void addFullyConnectedLayer(size_t, ActivationFunction);
  void addClassificationLayer(size_t);
  
  void initialize();

  // unused function, remove later
  // void setParameters(size_t, const Matrix&, const Vector&, ActivationFunction);

  Vector eval(const DataArray&) const;

  void train(const std::vector<TrainingData>&, OptimizationOptions options=OptimizationOptions());

  // functional evaluation (for training routine)
  double evalFunctional(const std::vector<TrainingData>&,
			 std::vector<std::vector<DataArray*>>&,
			 std::vector<std::vector<DataArray*>>&,
			 const std::vector<size_t>&,
			 OptimizationOptions) const;
  
  // gradient evaluation (for training routine)
  NeuralNetwork evalGradient(const std::vector<TrainingData>&,
			      const std::vector<std::vector<DataArray*>>&,
			      const std::vector<std::vector<DataArray*>>&,
			      const std::vector<size_t>&,
			      OptimizationOptions) const;

  double dot(const NeuralNetwork&) const;
  double norm() const;  

  // operator overloads
  NeuralNetwork& operator*=(double);
  NeuralNetwork& operator+=(const ScaledNeuralNetwork&);
  friend ScaledNeuralNetwork operator*(double, const NeuralNetwork&);      
  friend NeuralNetwork operator+(const NeuralNetwork&, const NeuralNetwork&);  
  friend NeuralNetwork operator+(const NeuralNetwork& lhs, const ScaledNeuralNetwork& rhs);
    
  // console output
  friend std::ostream& operator<<(std::ostream&, const NeuralNetwork&);
  
 private:
  // Store dimension of neural network
  std::vector<Layer> layers;  
  
  bool initialized;
  
  // Weights and biases (old code, now parameters stored in vector "layers")
  // NeuralNetworkParameters params;

  // Random number generator
  std::mt19937 rnd_gen;
  std::uniform_real_distribution<> random_real;
  
  // for debugging and testing
  void gradientTest(const NeuralNetwork&,
		    const std::vector<TrainingData>&,
		    const std::vector<size_t>&,
		    OptimizationOptions) const;
};

#endif
