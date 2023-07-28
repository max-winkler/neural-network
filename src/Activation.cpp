#include "Activation.h"

#include <algorithm>

double activate(double x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return x;
    case ActivationFunction::SIGMOID:
      return 1./(1.+exp(-x));
    case ActivationFunction::TANH:
      return 2./(1.+exp(-2.*x)) - 1.;
    case ActivationFunction::RELU:
      return std::max(0., x);
    }
  
  return 0.;
}

Vector activate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  size_t n = x.size();
  Vector y(n);

  if(act == ActivationFunction::SOFTMAX)
    {
      double sum = 0.;
      for(size_t i=0; i<n; ++i)
        sum += exp(x[i]);
      for(size_t i=0; i<n; ++i)
        y[i] = exp(x[i])/sum;
    }
  else
    {
      // Component-wise applied activation functions
      for(size_t i=0; i<n; ++i)    
        y[i] = activate(x[i], act);
    }
  return y;
}

double Dactivate(double x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return 1.;
    case ActivationFunction::SIGMOID:
      return exp(-x)/pow(1.+exp(-x), 2.);
    case ActivationFunction::TANH:
      return 4.*exp(-2.*x)/pow(1.+exp(-2.*x), 2.);
    case ActivationFunction::RELU:
      return x<=0. ? 0. : 1.;
    }
  
  return 0.;
}

Vector Dactivate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  size_t n = x.size();
  Vector y(n);

  for(size_t i=0; i<n; ++i)    
    y[i] = Dactivate(x[i], act);

  return y;
}

Matrix DactivateCoupled(const Vector& x, ActivationFunction act)
{
  size_t n = x.size();
  Matrix J(n, n);

  // TODO: Assemble the fucking matrix here
  double e_sum = 0.;
  for(size_t i=0; i<n; ++i)
    e_sum += exp(x[i]);

  for(size_t i=0; i<n; ++i)
    for(size_t j=0; j<n; ++j)
      J[i][j] = (exp(x[i]) / e_sum) * ((i==j ? 1. : 0) - (exp(x[j]) / e_sum));

  return J;	       
}
