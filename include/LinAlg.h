#ifndef _LINALG_H_
#define _LINALG_H_

#define POOLING_MAX 0
#define POOLING_AVG 1

#include "Matrix.h"
#include "Tensor.h"

// Forward declarations
class MatrixView;

namespace linalg{
  /**
   * Computes the element-wise product of two matrices. Both matrices must have the same size.
   *
   * @brief Element-wise multiplication (Hadamard product)
   *
   * @param A Matrix on left-hand side of multiplication.
   * @param B Matrix on right-hand side of multiplication.
   */
  Matrix multiply(const MatrixView&, const MatrixView&);

  /**
   * Computes the Frobenius inner product of two matrices. Both matrices must have the same size.
   *
   * @brief Frobenius inner product
   *
   * @param A Matrix on the left-hand side of multiplication.
   * @param B Matrix on the right-hand side of multiplication.
   */
  float dot(const MatrixView&, const MatrixView&);

  /**
   * Computes convolution of the matrix with the kernel matrix \p K. Without specifying the
   * default parameters a stride of 1 no padding is used. The size of the resulting matrix is
   * determined automatically.
   *
   * @brief Convolution with kernel matrix
   *
   * @param A The matrix on the left-hand side of the convolution operation.
   * @param K The kernel matrix for the convolution. It is assumed that \p K is a square matrix.
   * @param S The stride parameter for the convolution, this is, the shift applied to the kernel 
   * in the convolution.
   * @param P The padding parameter for the convolution, this is, the matrix (*this) is extended by
   * \p P leading and conclusive zero rows and columns.
   * @param flip Is false when cross-correlation, and true when the mathematical convolution
   * operation shall be applied. 
   */
  Matrix convolve(const MatrixView&, const MatrixView&,
	        size_t S=1, size_t P=0, bool flip=false);
  Matrix tensor_convolve(const Tensor&, const Tensor&,
		     size_t S=1, size_t P=0, bool flip=false);
  Tensor tensor_convolve(const Tensor&, const Matrix&,
		     size_t, size_t);

  /**
   * Pooling operation for an input matrix. Produces a smaller matrix summarizing a batch
   * of picels of the original matrix. Implemented is max pooling only.
   *
   * @brief Pooling of a matrix view
   *
   * @param A The input matrix to which pooling should be applied.
   * @param type The type of pooling that should be done. Implemented are
   *        POOLING_MAX for max pooling and (soon) POOLING_AVG for average pooling.
   * @param S The stride parameter for the pooling operation (default is 2).
   * @param P The padding to be added to the original matrix before pooling.
   */
  Matrix pool(const MatrixView&, int type=POOLING_MAX, size_t S=2, size_t P=0);

  /**
   * Operation that returns the gradient of the pooling operation with respect to the input matrix.
   * 
   * @brief Gradient of the pooling operation.
   *
   * @param A I forgot which matrix this was. See PoolingLayer::backward_propagate(...).
   * @param B The original matrix the pooling was applied to.
   * @param type The pooling type (POOLING_MAX or POOLING_AVG).
   * @param S The stride parameter used for the pooling operation.
   * @param P The padding that was added to the original matrix before pooling.
   */
  Matrix unpool(const MatrixView&, const MatrixView&, int type=POOLING_MAX, size_t S=2, size_t P=0);
}

/**
 * Class representing the view on a matrix. This can stem e.g. from a TensorSlice or a Matrix itself.
 */
class MatrixView
{
public:
  MatrixView(const float*, size_t, size_t);
  MatrixView(const Matrix&);
  MatrixView(const TensorSlice&);
  
  size_t nRows() const;
  size_t nCols() const;

  inline float operator()(size_t i, size_t j) const{
    return data[i*n + j];
  }  

  // TODO: Let multiply return a proxy object "HadamardView" to avoid memory allocation
  friend Matrix linalg::multiply(const MatrixView&, const MatrixView&);
  friend float linalg::dot(const MatrixView&, const MatrixView&);
  
private:
  const float* data;
  size_t m;
  size_t n;
};

#endif
