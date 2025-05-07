#ifndef _MATRIX_H_
#define _MATRIX_H_

#define POOLING_MAX 0
#define POOLING_AVG 1

#include <iostream>

#include "DataArray.h"

// Forward class declarations
class Vector;
class Rank1Matrix;
class MatrixView;
class Matrix;
struct ScaledMatrix;

// Forward declarations of free functions
ScaledMatrix operator*(double, const Matrix&);


/**
 * Proxy class representing a matrix row. This is used to allow element access of matrices 
 * using [i][j] notation. Each instance holds a pointer to the i-th row of a matrix and allows
 * access to an entry using operator[].
 */
class MatrixRow
{
 public:
  /**
   * Constructor creating matrix row proxy represented by a pointer to the beginning of
   * the matrix row.
   *
   * @param row_ptr Pointer to the first entry of the matrix row. 
   */
  MatrixRow(double*);

  /**
   * Access single element of a matrix row by reference for writing.
   *
   * @param i Index of the element to be accessed.
   */
  double& operator[](size_t);

  /**
   * Access single element of a matrix row by const reference for reading.
   *
   * @param i Index of the element to be accessed.
   */
  const double& operator[](size_t) const;
  
private:
  /**
   * Pointer to the first element of the matrix row.
   */
  double* data_ptr;
};

/**
 * Class implements a matrix data structure and provides basic routines for matrix operations.
 */
class Matrix : public DataArray
{
 public:
  /**
   * Default constructor creating matrix of dimension 0x0.
   *
   * @brief Default constructor
   */
  Matrix();

  /**
   * Constructor creating a zero matrix of size \p m x \p n.
   *
   * @brief Create matrix with zero entries
   *
   * @param m Number of rows of the matrix.
   * @param n Number of columns of the matrix.
   */
  Matrix(size_t, size_t);

  /**
   * Constructor creating a matrix with a certain size and copying the values the poiner \p ptr points to.
   * This assumes that \p ptr points to a contiguous block of \p m x \p n double values.
   *
   * @brief Create matrix and initialize by values from given data block
   *
   * @param m Number of rows of the matrix.
   * @param n Number of columns of the matrix.
   * @param ptr Pointer to the data block to be copied into the matrix.
   */
  Matrix(size_t, size_t, const double*);  

  /**
   * Copy constructor creating a hard copy of another matrix.
   *
   * @brief Copy constructor
   *
   * @param A The matrix to be copied.
   */
  Matrix(const Matrix&);
  // TODO: Implement move constructor

  /**
   * Constructor creating a matrix of size \p m x \p n and initializing their values with image data 
   * behind the pointer \p ptr. It is assumed that \p ptr points at a contiguous block of unsigned chars.
   * The unsigned chars (0-255) are converted to double precision values in the range 0.0-1.0. 
   * This constructor is helpful when image data should be processed in the neural network.
   *
   * @brief Create matrix from image data
   *
   * @param m The number of rows of the matrix.
   * @param n The number of columns of the matrix.
   * @param ptr A contiguous block of \p m x \p n unsigned chars representing image data.
   */ 
  Matrix(size_t, size_t, const unsigned char*);

  /**
   * Copy assignment operator creating a hard copy of another matrix.
   *
   * @brief Copy assignment operator
   *
   * @param other The matrix to be copied.
   */
  Matrix& operator=(const Matrix&);

  /**
   * Move assignment operator used to move a matrix coming as rvalue and avoid copying.
   *
   * @brief Move assignment operator
   *
   * @param other The matrix to be moved.
   */
  Matrix& operator=(Matrix&&);

  /**
   * Assign entries from a brace-enclosed list of values.
   * 
   * @brief Assignment by brace-enclosed list.
   *
   * @param list The entry list to be assigned to the current matrix. It is assumed that the 
   * number of elements in \p list is equal to \p m x \p n. 
   */
  Matrix& operator=(std::initializer_list<double>);
  
  /**
   * Return the number of rows of the matrix.
   *
   * @brief Get number of rows.
   */
  size_t nRows() const;

  /**
   * Return the number of columns of the matrix.
   *
   * @brief Get number of columns.
   */
  size_t nCols() const;
  
  /**
   * Access a row of the matrix for writing.
   * This method returns an instance of the proxy class MatrixRow 
   * which can be accessed again via operator[].
   *
   * @brief Access matrix row.
   *
   * @param i Index of the row to be accessed.
   */
  MatrixRow operator[](size_t);

  /**
   * Access a row of the a const matrix for reading. 
   * This method returns an instance of the proxy class MatrixRow 
   * which can be accessed again via operator[].
   *
   * @brief Access matrix row.
   *
   * @param i Index of the row to be accessed.
   */
  const MatrixRow operator[](size_t) const;  

    /**
   * Access single element using the notation A(i,j), which is more efficient that A[i][j].
   *
   * @param i Row index of the element to be accessed.
   * @param j Column index of the element to be accessed.
   */
  double& operator()(size_t, size_t);

  /**
   * Access single element using the notation A(i,j), which is more efficient that A[i][j].
   *
   * @param i Row index of the element to be accessed.
   * @param j Column index of the element to be accessed.
   */
  const double& operator()(size_t, size_t) const;
  
  /**
   * Multiplies all matrix entries with a given value.
   *
   * @brief Scale matrix
   *
   * @param a The value to multiply to each matrix entry.
   */
  Matrix& operator*=(double);

  /**
   * Computes the matrix-vector product.
   *
   * @brief Matrix-vector product
   *
   * @param x The vector to be multiplied with (*this)
   */
  Vector operator*(const Vector&) const;

  /**
   * Adds another matrix to the current one and returns the resulting matrix.
   * Matrices have the same size, otherwise an empty matrix is returned.
   *
   * @brief Adding two matrices
   *
   * @param x The matrix to be added with (*this)
   */
  Matrix operator+(const Matrix&) const;

  /**
   * Adds a value to each entry of the matrix.
   *
   * @brief Adding value to each element of the matrix.
   *
   * @param x Value to be added to each matrix element.
   */  
  Matrix& operator+=(double);

  /**
   * Adds another matrix to the current one. Both matrices must have the same size.
   *
   * @brief Add another matrix
   *
   * @param A The matrix to be added to (*this)
   */
  Matrix& operator+=(const Matrix&);

  /**
   * Adds a scaled matrix a*X to the current one. The matrix X must have the correct size.
   * ScaledMatrix is a proxy class representing the result of \ref operator*(double, const Matrix&) 
   * without storing this matrix directly.
   *
   * @brief Add a multiple of another matrix
   *
   * @param B Scaled matrix to be added to (*this).
   */
  Matrix& operator+=(const ScaledMatrix&);

  /**
   * Adds an instance of Rank1Matrix to the current one. An instance of Rank1Matrix is for instance created
   * as result of the \ref function outer(const Vector&, const Vector&).
   *
   * @brief Add a Rand1Matrix
   *
   * @param A Rank 1 matrix to be added to (*this)
   */
  Matrix& operator+=(const Rank1Matrix&);
  
  /**
   * Flattens the matrix to a vector. The entries of the matrix are appended row-wise to the vector.
   *
   * @brief Flatten matrix to vector
   */
  Vector flatten() const;


  /**
   * Adjoint operation to the convolution. This method returns the gradient of the operation A.convolve(K)
   * with respect to the kernel matrix K.
   *
   * @brief Get gradient with of convolution with respect to kernel matrix
   *
   * @param Y The matrix representing the gradient up to the previous layer.
   * @param J Parameter indicating how many entries are skipped. This corresponds to the stride in the forward operation.
   * @param P The padding parameter.
   */
  Matrix back_convolve(const Matrix&, size_t J=1, size_t P=0) const;

  /**
   * Pooling operation of a matrix. Produces a smaller matrix summarizing a batch of pixels of the original matrix.
   *
   * @brief Pooling of a matrix
   *
   * @param type The type of pooling that should be done. Implemented are POOLING_MAX for max pooling and POOLING_AVG
   * for average pooling.
   * @param S The stride parameter for the pooling operation (default is 2).
   * @param P The padding to be added to the original matrix before pooling.
   */
  Matrix pool(int type=POOLING_MAX, size_t S=2, size_t P=0) const;

  /**
   * Operation that returns the gradient of the pooling operation with respect to the input matrix.
   * 
   * @brief Gradient of the pooling operation.
   *
   * @param A The original matrix the pooling was applied to.
   * @param type The pooling type (POOLING_MAX or POOLING_AVG).
   * @param S The stride parameter used for the pooling operation.
   * @param P The padding that was added to the original matrix before pooling.
   */
  Matrix unpool(const Matrix&, int type=POOLING_MAX, size_t S=2, size_t P=0) const;

  /**
   * Computes the Kronecker product of two matrices. This method is more general than the original Kronecker product
   * and allows overlaps and gaps. This depends on stride parameter and size of the kernel matrix. In machine learning 
   * this is the operation required to compute the gradient of a convolution with respect to the input matrix.
   *
   * @brief Generalized Kronecker product of two matrices.
   *
   * @param K The kernel matrix.
   * @param S The stride parameter (default value is the size of the kernel matrix \p K)
   * @param P The padding parameter used in the original convolution operation.
   */
  Matrix kron(const Matrix&, int S=0, int overlap=0) const;  

  /**
   * Writes a matrix with values between 0 and 1 into an array of unsigned chars representing the pixel values of an 
   * image. This operation allows to write matrices representing images into a file.
   *
   * @param pixels Pointer to the pixel values that should be written.
   */
  void write_pixels(unsigned char*) const;
  
  /**
   * Output stream operator. Used to write the matrix to console or to a file.
   *
   * @brief Vector output
   * 
   * @param os The output stream.
   * @param matrix The matrix to be printed.
   */
  friend std::ostream& operator<<(std::ostream&, const Matrix&);

  // Friend declarations
  friend class Vector;
  friend class MatrixView;
  //friend Matrix linalg::multiply(const MatrixView&, const MatrixView&);
  
private:
  /**
   * Number of rows of the matrix.
   */
  size_t m;
};

/**
 * Class represents the result of the opetration (double)*(Matrix). The result is computed on-the-fly at element access 
 * and never computed at once. This proxy class is used to save computational cost and memory allocations.
 */
struct ScaledMatrix
{
  /**
   * Constructor creating a ScaledMatrix object for given scale factor and reference to a matrix object.
   *
   * @brief Constructor initializing a ScaledMatrix object.
   *
   * @param scale The scale factor.
   * @param matrix A reference to the matrix to be scaled.
   */
  ScaledMatrix(double, const Matrix& matrix);
  double scale;
  const Matrix* matrix;
};

#endif
