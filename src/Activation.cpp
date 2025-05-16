#include "Activation.h"

#include <algorithm>

float activate(float x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return x;
    case ActivationFunction::SIGMOID:
      return x>0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
    case ActivationFunction::TANH:
      return 2.0f / (1.0f + exp(-2.0f*x)) - 1.0f;
    case ActivationFunction::RELU:
      return std::max(0.0f, x);
    }
  
  return 0.;
}

Vector activate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  Vector y(x.length());

  if(act == ActivationFunction::SOFTMAX)
    {
      float sum = 0.0f;
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

float Dactivate(float x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  switch(act)
    {
    case ActivationFunction::NONE:
      return 1.0f;
    case ActivationFunction::SIGMOID:
      {
        float sig = 1.0f / (1.0f + exp(-x));
        return sig * (1.0f - sig);
      }
    case ActivationFunction::TANH:
      return 4.0f*exp(-2.0f*x)/pow(1.+exp(-2.0f*x), 2.0f);
    case ActivationFunction::RELU:
      return x<0.0f ? 0.0f : 1.0f;
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

  float e_sum = 0.;
  for(size_t i=0; i<n; ++i)
    e_sum += exp(x[i]);

  if(e_sum < 1.e-6f)
    {
      std::cerr << "WARNING: Division by zero in Jacobi matrix of SOFTMAX activation function.\n";
    }
  
  for(size_t i=0; i<n; ++i)
    for(size_t j=0; j<n; ++j)
      J[i][j] = (exp(x[i]) / e_sum) * ((i==j ? 1.0f : 0.0f) - (exp(x[j]) / e_sum));

  return J;	       
}
