#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include <vector>

#include "Matrix.h"
#include "Vector.h"
#include "Activation.h"

typedef std::vector<size_t> Dimension;

class NeuralNetwork {
 public:
  NeuralNetwork(Dimension);

  void setParameters(size_t, const Matrix&, const Vector&, ActivationFunction);

  double eval(const Vector&) const;
  
  friend std::ostream& operator<<(std::ostream&, const NeuralNetwork&);
  
 private:
  // Store dimension of neural network
  std::vector<size_t> width;
  size_t layers;

  // Weights and biases
  std::vector<Matrix> weight;
  std::vector<Vector> bias;
  std::vector<ActivationFunction> activation;
};

#endif
