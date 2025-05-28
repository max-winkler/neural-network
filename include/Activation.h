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
float activate(float, ActivationFunction);
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
