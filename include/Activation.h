#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>

#include "Vector.h"

enum ActivationFunction
  {
    NONE, SIGMOID, TANH
  };


double activate(double x, ActivationFunction act);
Vector activate(const Vector& x, ActivationFunction act);
double Dactivate(double x, ActivationFunction act);
Vector Dactivate(const Vector& x, ActivationFunction act);

#endif
