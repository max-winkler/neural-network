#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>

#include "Vector.h"
#include "Matrix.h"

enum ActivationFunction
  {
    NONE, SIGMOID, TANH, RELU, SOFTMAX
  };


double activate(double, ActivationFunction);
Vector activate(const Vector&, ActivationFunction);
Matrix activate(const Matrix&, ActivationFunction);

double Dactivate(double, ActivationFunction);
Vector Dactivate(const Vector&, ActivationFunction);
Matrix Dactivate(const Matrix&, ActivationFunction);
Matrix DactivateCoupled(const Vector&, ActivationFunction);

#endif
