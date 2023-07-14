#include "Activation.h"

double activate(double x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return x;
    case ActivationFunction::SIGMOID:
      return 1./(1+exp(-x));
    case ActivationFunction::TANH:
      return 2/(1+exp(-2*x)) - 1;
    }
  
  return 0.;
}

Vector activate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  size_t n = x.size();
  Vector y(n);

  for(size_t i=0; i<n; ++i)    
    y[i] = activate(x[i], act);

  return y;
}
