#include "Activation.h"

#include <algorithm>

double activate(double x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return x;
    case ActivationFunction::SIGMOID:
      return x>0. ? 1./(1.+exp(-x)) : exp(x)/(1+exp(x));
    case ActivationFunction::TANH:
      return 2./(1.+exp(-2.*x)) - 1.;
    case ActivationFunction::RELU:
      return std::max(0., x);
    }
  
  return 0.;
}

Vector activate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  Vector y(x.length());

  if(act == ActivationFunction::SOFTMAX)
    {
      double sum = 0.;
      for(size_t i=0; i<x.length(); ++i)
        sum += exp(x[i]);
      for(size_t i=0; i<x.length(); ++i)
        y[i] = exp(x[i])/sum;
    }
  else
    {
      // Component-wise applied activation functions
      for(size_t i=0; i<x.length(); ++i)    
        y[i] = activate(x[i], act);
    }
  return y;
}

Matrix activate(const Matrix& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  Matrix y(x.nRows(), x.nCols());

  if(act == ActivationFunction::SOFTMAX)
    {
      std::cerr << "ERROR: Softmax activation function not implemented for matrix-valued layers.\n";
      return Matrix(0,0);
    }

  // Component-wise applied activation functions
  for(size_t i=0; i<x.nRows(); ++i)
    for(size_t j=0; j<x.nCols(); ++j)    
      y(i,j) = activate(x(i,j), act);
  
  return y;
}

double Dactivate(double x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return 1.;
    case ActivationFunction::SIGMOID:
      return x>0. ? exp(-x)/pow(1.+exp(-x), 2.) : exp(x)/pow(1+exp(x), 2.);
    case ActivationFunction::TANH:
      return 4.*exp(-2.*x)/pow(1.+exp(-2.*x), 2.);
    case ActivationFunction::RELU:
      return x<0. ? 0. : 1.;
    }
  
  return 0.;
}

Vector Dactivate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  Vector y(x.length());

  for(size_t i=0; i<x.length(); ++i)    
    y[i] = Dactivate(x[i], act);

  return y;
}

Matrix Dactivate(const Matrix& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  Matrix y(x.nRows(), x.nCols());

  // Component-wise applied activation functions
  for(size_t i=0; i<x.nRows(); ++i)
    for(size_t j=0; j<x.nCols(); ++j)    
      y(i,j) = Dactivate(x(i,j), act);
  
  return y;
}

Matrix DactivateCoupled(const Vector& x, ActivationFunction act)
{
  size_t n = x.nEntries();
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
