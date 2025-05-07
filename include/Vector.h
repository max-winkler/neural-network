#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>

#include "Matrix.h"
#include "Tensor.h"

// Forward class declarations
class DiagonalMatrix;
class Rank1Matrix;
class Vector;
struct ScaledVector;

// Forward declaration of free functions
ScaledVector operator*(double, const Vector&);
double norm(const Vector&, double p=2.);
Rank1Matrix outer(const Vector&, const Vector&);

/**
 * Proxy class representing the result of a scalar multiplied to a vector. This class avoids direct computation
 * of the resulting matrix and instead stores the scale factor and a pointer to the matrix so that the elements
 * are computed at element access. An instance of this class is created by the function 
 * \r operator*(double, const Vector&).
 */
struct ScaledVector
{
  /**
   * Constructor taking the scale factor and a const reference to a vector storing both as member variables.
   *
   * @param scale The scale factor.
   * @param x The vector to be scaled.
   */
  ScaledVector(double, const Vector&);

  /**
   * The scale factor.
   */
  double scale;
  /**
   * Pointer to the vector to be multiplied with the scale factor.
   */
  const Vector* vector;
};

/**
 * Class implements a vector data structure and provides basic routines for vector operations.
 */
class Vector : public DataArray {
 public:
  /**
   * Constructor initializing an empty vector.
   *
   * @brief Default constructor
   */
  Vector();

  /**
   * Constructor creating a vector of size \p size and setting all entries to zero.
   *
   * @brief Constructor initializing a vector with zero values.
   *
   * @param size The size of the vector.
   */
  Vector(size_t);

  /**
   * Constructor creating a vector with size \p size and initializing the values with the values
   * the pointer \p data is pointing to. It is assumed that \p data points to a consecutive block of 
   * \p size double precision values.
   *
   * @brief Create vector and initialize with values from a data block.
   *
   * @param size The size of the vector.
   * @param data A pointer to a block of doubles to be copied into the vector.
   */
  Vector(size_t, const double*);

  /**
   * Copy constructor creating a hard copy of another vector.
   *
   * @brief Copy constructor
   *
   * @param other The vector to be copied.
   */
  Vector(const Vector&);

  /**
   * Creates a vector from a list of values. The vector size is determined automatically.
   *
   * @brief Create vector from a list.
   *
   * @param list A list of values for the vector entries.
   */
  Vector(std::initializer_list<double>);

  /**
   * Destructor used to free the internal memory.
   *
   * @brief Desctructor
   */
  // ~Vector();
  
  /**
   * Copy assignment opertor creating a hard copy of another vector.
   *
   * @brief Copy assignment
   *
   * @param other The vector to be copied.
   */
  Vector& operator=(const Vector&);

  /**
   * Move assignment operator used to move a vector coming as rvalue. This avoids hard copying of the vector.
   *
   * @brief Move assignment operator
   *
   * @param other The vector to be moved.
   */
  Vector& operator=(Vector&&);

  /**
   * Assign entries from a brave-enclosed list of values to the vector. The vector size might change depending on
   * the size of the list.
   *
   * @brief Assignment by brace-enclosed list.
   *
   * @param list The list storing the values to be copied to the vector.
   */
  Vector& operator=(std::initializer_list<double>);
  
  /**
   * Summation of two vectors. Returns the resulting vector.
   *
   * @brief Vector summation
   *
   * @param x The vector to be added to (*this).
   */
  Vector operator+(const Vector&) const;

  /**
   * Difference of two vectors. Returns the resulting vector.
   *
   * @brief Vector difference
   *
   * @param x The vector to be subtracted from (*this).
   */  
  Vector operator-(const Vector&) const;

  /**
   * Adds another vector to the current one.
   *
   * @brief Add another vector
   *
   * @param x The vector to be added to (*this).
   */
  Vector& operator+=(const Vector&);

  /**
   * Adds another scaled vector a*x to the current one.
   *
   * @brief Add some multiple of another vector
   *
   * @param A The scaled vector to be added to (*this).
   */
  Vector& operator+=(const ScaledVector&);

  /**
   * Scale the vector by a scalar.
   *
   * @brief Vector scaling
   *
   * @param a The scale factor.
   */
  Vector& operator*=(double);

  /**
   * Returns the index where the vector attains its maximum value.
   *
   * @brief Get index of minimal entry   
   */
  size_t indMax() const;
  
  /**
   * Returns the length of the vector
   *
   * @brief Get vector length
   */
  size_t length() const;

  /**
   * Reshapes a vector with N entries to an \p m x \p n matrix. The parameters must be chosen to satisfy
   * N = m*n, otherwise a null vector is returned.
   *
   * @brief Reshape vector to a matrix
   *
   * @param m Number of rows of the resulting matrix.
   * @param m Number of columns of the resulting matrix.
   */
  Matrix reshape(size_t, size_t) const;

  /**
   * Reshapes a vector with N entries to an \p d x \p m x \p n tensor. The parameters must satisfy
   * N=d*m*n, otherwise a null tensor is returned.
   *
   * @brief Reshape vector to a tensor
   *
   * @param d Number of channels of the resulting tensor.
   * @param m Number of rows of the resulting tensor.
   * @param n Number of columns of the resulting tensor.
   */
  Tensor reshape(size_t, size_t, size_t) const;
    
  /**
   * Output stream operator. Used to write the vector to console or to a file.
   *
   * @brief Vector output
   * 
   * @param os The output stream.
   * @param x The vector to be printed.
   */
  friend std::ostream& operator<<(std::ostream&, const Vector&);

  friend class Matrix;
  friend class DiagonalMatrix;
  
  // Matrix-vector operations
  friend Vector Matrix::operator*(const Vector&) const;
  Vector operator*(const Matrix&) const;
  Vector operator*(const DiagonalMatrix&) const;
  
  // Vector operations operations
  friend Rank1Matrix outer(const Vector&, const Vector&);
  friend DiagonalMatrix diag(const Vector&);
  friend double norm(const Vector&, double);

private:
  
};

// Class representing a diagonal matrix
class DiagonalMatrix {
public:
  DiagonalMatrix(const Vector&); // What happens when we invoke this with an rvalue?

  friend class Vector;
private:
  const Vector* diagonal;
};

class Rank1Matrix {
public:
  Rank1Matrix(const Vector&, const Vector&);

  size_t nRows() const;
  size_t nCols() const;
  
  friend class Vector;
  friend class Matrix;
private:
  const Vector* u;
  const Vector* v;
};


#endif
