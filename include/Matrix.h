#ifndef _MATRIX_H_
#define _MATRIX_H_

#define POOLING_MAX 0
#define POOLING_AVG 1

#include <iostream>

#include "DataArray.h"

class Vector;
class Rank1Matrix;

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
  Matrix& operator+=(const Rank1Matrix&);

  // Convolution operations
  Matrix convolve(const Matrix&, size_t S=1, size_t P=0) const;
  Matrix pool(int, size_t S=2, size_t P=0) const;
 
  void write_pixels(unsigned char*) const;
  
  // Console output
  friend std::ostream& operator<<(std::ostream&, const Matrix&);
  
  friend class Vector;

  // TODO: I do not want this. Extend Matrix class so that this is not necessary
  friend class NeuralNetwork;
private:
  // number of rows
  size_t m;
};

#endif
