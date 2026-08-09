#include "Activation.h"

#include <algorithm>

const std::unordered_map<ActivationFunction, std::string> ActivationFunctionName = {
  {NONE, "none"},
  {SIGMOID, "sigmoid"},
  {TANH, "tanh"},
  {RELU, "relu"},
  {SOFTMAX, "softmax"},
};

const std::unordered_map<std::string, ActivationFunction> ActivationFunctionFromName = {
  {"none", NONE},
  {"sigmoid", SIGMOID},
  {"tanh", TANH},
  {"relu", RELU},
  {"softmax", SOFTMAX},
};

Vector activate(const Vector& x, ActivationFunction act=ActivationFunction::SIGMOID)
{
  Vector y(x.length());

  if(act == ActivationFunction::SOFTMAX)
    {
      float max_val = x[0];
      for(size_t i=1; i<x.length(); ++i)
        max_val = std::max(max_val, x[i]);

      float sum = 0.0f;
      for(size_t i=0; i<x.length(); ++i)
        {
          y[i] = exp(x[i]-max_val);
          sum += y[i];
        }
      for(size_t i=0; i<x.length(); ++i)
        y[i] /= sum;
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
      {
        float t = tanh(x);
        return 1.0f - t*t;
      }
    case ActivationFunction::RELU:
      return x < 0.0f ? 0.0f : 1.0f;
    default:
      std::cerr
          << "ERROR: Activation function can not be applied component-wise.\n";
      return 0.;      
    }
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
  Matrix J = Matrix::uninitialized(n, n);

  const Vector p = activate(x, ActivationFunction::SOFTMAX);

  for(size_t i=0; i<n; ++i)
    {
      const float p_i = p[i];
      for(size_t j=0; j<n; ++j)
        J(i,j) = p_i * ((i==j ? 1.0f : 0.0f) - p[j]);
    }

  return J;	       
}
