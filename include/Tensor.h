#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <iostream>

#include "DataArray.h"

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
   * @param d Depth of the tensor.
   * @param m Height of the tensor.
   * @param n Width of the tensor.
   */
  Tensor(size_t, size_t, size_t);

  /**
   * Copy constructor creating a hard copy of another tensor.
   *
   * @brief Copy constructor
   *
   * @param T The tensor to be copied.
   */
  Tensor(const Tensor&);

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
   * @ brief Multiplication with scalar
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
   * Computes the sum of two tensors. Tensors are assumed to have the same shape,
   * otherwise an empty tensor ist returned.
   *
   * @brief Adding two tensors.
   *
   * @param T The tensor to be added to (*this).
   */
  Tensor operator+(const Tensor&) const;
  
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
