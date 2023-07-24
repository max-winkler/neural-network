#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>

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

class Matrix
{
 public:
  // Constructor and destructor
  Matrix();
  Matrix(size_t, size_t);
  Matrix(const Matrix&);
  ~Matrix();
  Matrix& operator=(const Matrix&);
  Matrix& operator=(Matrix&&);
  Matrix& operator=(std::initializer_list<double>);
    
  // Simple getter functions
  std::pair<size_t, size_t> size() const;
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
  Matrix& operator+=(const Matrix&);
  Matrix& operator+=(const Rank1Matrix&);
  
  // Console output
  friend std::ostream& operator<<(std::ostream&, const Matrix&);
  
  friend class Vector;
 private:
  // Dimension of matrix
  size_t m, n;
  // Vector for matrix entries
  double* data;
};

#endif
