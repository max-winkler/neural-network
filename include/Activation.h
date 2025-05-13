#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>

#include "Vector.h"
#include "Matrix.h"

enum ActivationFunction
  {
    NONE, SIGMOID, TANH, RELU, SOFTMAX
  };


float activate(float, ActivationFunction);
Vector activate(const Vector&, ActivationFunction);
Matrix activate(const Matrix&, ActivationFunction);

float Dactivate(float, ActivationFunction);
Vector Dactivate(const Vector&, ActivationFunction);
Matrix Dactivate(const Matrix&, ActivationFunction);
Matrix DactivateCoupled(const Vector&, ActivationFunction);

#endif
