#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>
#include <unordered_map>

#include "Vector.h"
#include "Matrix.h"

enum ActivationFunction
  {
    NONE, SIGMOID, TANH, RELU, SOFTMAX
  };

const static std::unordered_map<ActivationFunction, std::string> act_to_str = {
  {NONE, "none"},
  {SIGMOID, "sigmoid"},
  {TANH, "tanh"},
  {RELU, "relu"},
  {SOFTMAX, "softmax"},
};

float activate(float, ActivationFunction);
Vector activate(const Vector&, ActivationFunction);
Matrix activate(const Matrix&, ActivationFunction);

float Dactivate(float, ActivationFunction);
Vector Dactivate(const Vector&, ActivationFunction);
Matrix Dactivate(const Matrix&, ActivationFunction);
Matrix DactivateCoupled(const Vector&, ActivationFunction);

#endif
