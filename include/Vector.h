#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>

#include "Matrix.h"

// Forward declarations
class DiagonalMatrix;
class Rank1Matrix;

class Vector {
 public:
  // Constructor and destructor
  Vector();
  Vector(size_t);
  Vector(const Vector&);
  Vector(std::initializer_list<double>);
  ~Vector();

  Vector& operator=(const Vector&);
  Vector& operator=(Vector&&);
  Vector& operator=(std::initializer_list<double>);
  // Simple getter functions
  size_t size() const;

  // Array subscription operator
  double& operator[](size_t);
  const double& operator[](size_t) const;

  // Vector operations
  Vector operator+(const Vector&) const;
  Vector operator-(const Vector&) const;
  Vector& operator+=(const Vector&);
  Vector& operator*=(double);
  
  // Console output via output stream
  friend std::ostream& operator<<(std::ostream&, const Vector&);

  // friend classes
  friend class Matrix;
  friend class DiagonalMatrix;
  
  // Matrix-vector operations
  friend Vector Matrix::operator*(const Vector&) const;
  Vector operator*(const Matrix&) const;
  Vector operator*(const DiagonalMatrix&) const;
  
  // free functions
  friend Rank1Matrix outer(const Vector&, const Vector&);
  friend DiagonalMatrix diag(const Vector&);
  friend double norm(const Vector&, double p=2.);
 private:
  
  double* data;
  size_t n;
};

// Class representing a diagonal matrix
class DiagonalMatrix {
public:
  DiagonalMatrix(const Vector&);

  friend class Vector;
private:
  size_t n;
  const Vector* diagonal;
};

class Rank1Matrix {
public:
  Rank1Matrix(const Vector&, const Vector&);

  friend class Vector;
  friend class Matrix;
private:
  size_t m, n;
  const Vector* u;
  const Vector* v;
};


#endif
