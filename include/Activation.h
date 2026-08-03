#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>
#include <unordered_map>

#include "Vector.h"
#include "Matrix.h"

/// Enumeration for all implemented activation functions
enum ActivationFunction
  {
    NONE, SIGMOID, TANH, RELU, SOFTMAX
  };

/**
 * Mapping from activation function type to their string representation.
 */
extern const std::unordered_map<ActivationFunction, std::string> ActivationFunctionName;
/**
 * Mapping from activation function name to their type (enum ActivationFunction).
 */
extern const std::unordered_map<std::string, ActivationFunction> ActivationFunctionFromName;

/**
 * Apply activation function to single float value.
 */
inline float activate(float x, ActivationFunction act=SIGMOID)
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
    default:
      std::cerr << "ERROR: Unable to evaluate activation function component-wise.";
      return 0.;
    }
  
  return 0.0f;
}

/**
 * Apply activation function componentwise to a vector.
 */
Vector activate(const Vector&, ActivationFunction);
/**
 * Apply activation function componentwise to a matrix.
 */
Matrix activate(const Matrix&, ActivationFunction);

/**
 * Apply derivative of activation function to a single float value.
 */
float Dactivate(float, ActivationFunction);
/**
 * Apply derivative of activation function component-wise to a vector.
 */
Vector Dactivate(const Vector&, ActivationFunction);
/**
 * Apply derivative of activation function component-wise to a matrix.
 */
Matrix Dactivate(const Matrix&, ActivationFunction);
/**
 * Apply derivative of a fully-coupled activation function (in our case SOFTMAX).
 * This returns the Jacobian which is in this case not a diagonal matrix.
 */
Matrix DactivateCoupled(const Vector&, ActivationFunction);

#endif
