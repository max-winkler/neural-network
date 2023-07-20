#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <vector>

#include "Matrix.h"
#include "Vector.h"
#include "Activation.h"
#include "TrainingData.h"

typedef std::vector<size_t> Dimension;

struct ScaledNeuralNetworkParameters;

struct NeuralNetworkParameters
{  
  std::vector<Matrix> weight;
  std::vector<Vector> bias;
  std::vector<ActivationFunction> activation;

  double dot(const NeuralNetworkParameters&) const;
  
  // why friend? all attributes are public
  friend ScaledNeuralNetworkParameters operator*(double, const NeuralNetworkParameters&);
  // implement later if needed
  //friend NeuralNetworkParameters operator+(const NeuralNetworkParameters&, const NeuralNetworkParameters&); 
  friend NeuralNetworkParameters operator+(const NeuralNetworkParameters&, const ScaledNeuralNetworkParameters&);
  friend std::ostream& operator<<(std::ostream&, const NeuralNetworkParameters&);
};

struct ScaledNeuralNetworkParameters
{
  ScaledNeuralNetworkParameters(double, const NeuralNetworkParameters&);
  double scale;
  const NeuralNetworkParameters* params;
};

class NeuralNetwork
{
 public:
  NeuralNetwork(Dimension);

  void setParameters(size_t, const Matrix&, const Vector&, ActivationFunction);

  double eval(const Vector&) const;

  void train(const std::vector<TrainingData>&);

  // functional evaluation (for training routine)
  double eval_functional(const NeuralNetworkParameters&,
		     const std::vector<TrainingData>&,
		     std::vector<std::vector<Vector>>&,
		     std::vector<std::vector<Vector>>&) const;

  // gradient ecaluation (for training routine)
  NeuralNetworkParameters eval_gradient(const NeuralNetworkParameters&,
				const std::vector<TrainingData>&,
				const std::vector<std::vector<Vector>>&,
				const std::vector<std::vector<Vector>>&) const;
    
  friend std::ostream& operator<<(std::ostream&, const NeuralNetwork&);
  
 private:
  // Store dimension of neural network
  std::vector<size_t> width;
  size_t layers;

  // Weights and biases  
  NeuralNetworkParameters params;

  // for debugging and testing
  void gradient_test(const NeuralNetworkParameters&, const std::vector<TrainingData>&) const;
};

#endif
