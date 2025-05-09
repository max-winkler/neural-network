#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <iostream>

#include "DataArray.h"
#include "Matrix.h"

// forward declarations
class Vector;
class Tensor;
class ScaledTensor;

// forward declaration of free functions
ScaledTensor operator*(double, const Tensor&);

struct ScaledTensor
{
  /**
   * Constructor taking the scale factor and a const reference to a tensor storing both as
   * member variables.
   *
   * @param scale The scale factor.
   * @param x The tensor to be scaled.
   */
  ScaledTensor(double, const Tensor&);

  /**
   * The scale factor.
   */
  double scale;
  /**
   * Pointer to the vector to be multiplied with the scale factor.
   */
  const Tensor* tensor;
};

/**
 * Proxy class used to view a slice of a tensor and assign from or save into a matrix.
 */
class TensorSlice
{
  friend class Tensor;
  friend class MatrixView;
public:
  /**
   * Write a matrix into a tensor slice.
   *
   * @param A The matrix to be written into the slice.
   */
  TensorSlice& operator=(const Matrix&);
  
  /**
   * Add some number to each element of the tensor slice.
   *
   * @param a The number to be added to the tensor slice.
   */
  TensorSlice& operator+=(double);
  
  /**
   * Add a matrix to the tensor slice.
   *
   * @param A The matrix to be added.
   */
  TensorSlice& operator+=(const Matrix&);

  
  /**
   * Converts the tensor slice into a matrix view. MatrixView returned
   * by TensorSlice::to_matrix_view() is only valid as long as the original Tensor
   * exists and is unchanged.
   *
   * @brief Convert TensorSlice to MatrixView
   */
  MatrixView to_matrix_view() const;
  
private:
  /**
   * Constructor creating a matrix slice by specifying their dimension and providing a pointer
   * to the values.
   *
   * @brief Constructor
   *
   * @param m The number of rows of the tensor slice
   * @param n The number of columns of the tensor slice
   * @param data The pointer to the data of the tensor slice
   */
  TensorSlice(size_t, size_t, double*);

  /**
   * Access the elements of a matrix slice
   *
   * @param i The row index of the element to access.
   * @param j The columns index of the element to access.
   */
  double& operator()(size_t, size_t);

  /**
   * Number of rows of the tensor slice.
   */
  size_t m;

  /**
   * Number of columns of the tensor slice
   */
  size_t n;

  /**
   * Pointer to the data of the tensor slice (stored row-wise)
   */
  double* data;
};

/**
 * Class implements a stage-3 tensor data structure and provides basic routines for tensor operations
 */
class Tensor : public DataArray
{
 public:
  /**
   * Default constructor creating a 0x0x0 tensor.
   *
   * @brief Default constructor
   */
  Tensor();

  /**
   * Constructor creating a zero tensor of size \p d x \p m x \p n
   *
   * @brief Create 3-stage tensor with zero entries
   *
   * @param d Number of channels of the tensor.
   * @param m Number of rows of the tensor.
   * @param n Number of columns of the tensor.
   */
  Tensor(size_t, size_t, size_t);

  /**
   * Constructor creating a tensor of the specified size and copying the values from
   * a given data pointer.
   *
   * @brief Create 3-stage tensor from data pointer
   *
   * @param d Number of channels of the tensor.
   * @param m Number of rows of the tensor.
   * @param n Number of columns of the tensor.
   * @param x Pointer to the data.
   */
  Tensor(size_t, size_t, size_t, const double*);
  
  /**
   * Copy constructor creating a hard copy of another tensor.
   *
   * @brief Copy constructor
   *
   * @param T The tensor to be copied.
   */
  Tensor(const Tensor&);

  /**
   * Create a tensor of dimension 1 x m x n from a matrix of dimension m x n.
   *
   * @brief Copy constructor for matrix input
   *
   * @param A The matrix to be copied.
   */
  Tensor(const Matrix&);
  
  /**
   * Copy assignment operator creating a hard copy of another matrix.
   *
   * @brief Copy assignment operator
   *
   * @param other The tensor to be copied.
   */
  Tensor& operator=(const Tensor&);

  /**
   * Move assignment operator used to move a tensor coming as rvalue and avoid copying.
   *
   * @brief Move assignment operator
   *
   * @param other The tensor to be moved.
   */
  Tensor& operator=(Tensor&&);
  
  /**
   * Return the number of channels (depth) of the tensor.
   *
   * @brief Get number of channels.
   *
   */
  size_t nChannels() const;

  /**
   * Return the number of rows of the tensor.
   *
   * @brief Get number of rows.
   */
  size_t nRows() const;
  
  /**
   * Return number of columns of the tensor.
   *
   * @brief Get number of columns.
   */
  size_t nCols() const;

  /**
   * Access a single element using the notation T(c,i,j).
   *
   * @brief Element access
   *
   * @param c The index of the channel.
   * @param i The index of the row.
   * @param j The index of the column.
   */
  double& operator()(size_t, size_t, size_t);

  /**
   * Access a single element using the notation T(c,i,j) for constant T (read only).
   *
   * @brief Element access (const instance)
   *
   * @param c The index of the channel.
   * @param i The index of the row.
   * @param j The index of the column.   
   */
  const double& operator()(size_t, size_t, size_t) const;

  /**
   * Multiply a tensor with a scalar.
   *
   * @brief Multiplication with scalar
   *
   * @param a The value to multiply to each tensor entry.
   */
  Tensor& operator*=(double);

  /**
   * Adds another tensor to the current one. Tensors are assumed to have the same shape, otherwise
   * an empty tensor is returned.
   *
   * @brief Adding tensor to (*this).
   *
   * @param T The tensor to be added.
   */
  Tensor& operator+=(const Tensor&);

  /**
   * Adds a multiple of another tensor to the current one. Tensors must have the same shape,
   * otherwise an empty tensor is returned.
   *
   * @brief Add multiple of a tensor to (*this).
   *
   * @param S The scaled tensor, usually created by operator*(double, const Tensor&).
   */
  Tensor& operator+=(const ScaledTensor&);
  
  /**
   * Subtracts another tensor from the current one. Tensors are assumed to have the same shape,
   * otherwise an empty tensor is returned.
   *
   * @brief Subtracting tensor from (*this).
   *
   * @param T The tensor to be subtracted.
   */
  Tensor& operator-=(const Tensor&);
  
  /**
   * Computes the sum of two tensors. Tensors are assumed to have the same shape,
   * otherwise an empty tensor ist returned.
   *
   * @brief Adding two tensors.
   *
   * @param T The tensor to be added to (*this).
   */
  Tensor operator+(const Tensor&) const;

  /**
   * Compute the convolution with another tensor (the kernel).
   *
   * @brief Convolution of two tensors.
   *
   * @param K The kernel tensor for the convolution.
   * @param S The stride parameter.
   * @param P The padding parameter.
   * @param flip Is false when cross-correlation, and true when the mathematical convolution
   * shall be applied
   */
  Matrix convolve(const Tensor&, size_t S=1, size_t P=0, bool flip=false) const;

  /**
   * Compute the convolution with a matrix. Returns a tensor where the c-th channel
   * is the convolution of the c-th channel of the input and the matrix.
   *
   * @param K The kernel matrix for the convolution.
   * @param S The stride parameter.
   * @param P The passing parameter.
   */
  Tensor convolve(const Matrix&, size_t, size_t) const;
  
  /**
   * Flatten the tensor to a vector.
   *
   * @brief Flatten to a vector.
   */
  Vector flatten() const;
  
  /**
   * Create a matrix from a tensor by taking the slice. The first index (channel) is fixed.
   *
   * @param c The index of the channel the slice must go through.
   */
  TensorSlice operator[](size_t);
  TensorSlice operator[](size_t) const;
  
  
 private:
  /**
   * Depth of the tensor
   */
  size_t d;

  /**
   * Height of the tensor
   */
  size_t m;
};

#endif
