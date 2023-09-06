#ifndef _MATRIX_H_
#define _MATRIX_H_

#define POOLING_MAX 0
#define POOLING_AVG 1

#include <iostream>

#include "DataArray.h"

// Forward class declarations
class Vector;
class Rank1Matrix;
class Matrix;
struct ScaledMatrix;

// Forward declarations of free functions
ScaledMatrix operator*(double, const Matrix&);

class MatrixRow
{
 public:
  MatrixRow(double*);
  double& operator[](size_t);
  const double& operator[](size_t) const;
 private:
  double* data_ptr;
};

class Matrix : public DataArray
{
 public:
  // Standard constructors
  Matrix();
  Matrix(size_t, size_t);
  Matrix(size_t, size_t, const double*);  
  Matrix(const Matrix&);
  // TODO: Implement move constructor

  // Read matrix from image pixels
  Matrix(size_t, size_t, const unsigned char*);

  // Assignment operators
  Matrix& operator=(const Matrix&);
  Matrix& operator=(Matrix&&);
  Matrix& operator=(std::initializer_list<double>);
  
  // Simple getter functions
  size_t nRows() const;
  size_t nCols() const;
  
  // Element acces
  MatrixRow operator[](size_t);
  const MatrixRow operator[](size_t) const;  

  // Matrix operations
  Matrix& operator*=(double);

  // Matrix-vector operations
  Vector operator*(const Vector&) const; ;

  // Matrix-matrix operations
  Matrix operator+(const Matrix&) const;
  Matrix& operator+=(const Matrix&);
  Matrix& operator+=(const ScaledMatrix&);
  Matrix& operator+=(const Rank1Matrix&);

  // Convolution operations
  Vector flatten() const;
  Matrix convolve(const Matrix&, size_t S=1, size_t P=0) const;
  Matrix pool(int, size_t S=2, size_t P=0) const;
 
  void write_pixels(unsigned char*) const;
  
  // Console output
  friend std::ostream& operator<<(std::ostream&, const Matrix&);

  // Friend declarations
  friend class Vector;
  
private:
  // number of rows
  size_t m;
};

struct ScaledMatrix
{
  ScaledMatrix(double, const Matrix& matrix);
  double scale;
  const Matrix* matrix;
};

#endif
