#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>

#include "Matrix.h"

// Forward declarations
class DiagonalMatrix;
class Rank1Matrix;

double norm(const Vector&, double p=2.);
Rank1Matrix outer(const Vector&, const Vector&);

class Vector : public DataArray {
 public:
  // Constructor and destructor
  Vector();
  Vector(size_t);
  Vector(size_t, const double*); 
  Vector(const Vector&);
  Vector(std::initializer_list<double>);

  // Assignment operators
  Vector& operator=(const Vector&);
  Vector& operator=(Vector&&);
  Vector& operator=(std::initializer_list<double>);
  
  // Vector operations
  Vector operator+(const Vector&) const;
  Vector operator-(const Vector&) const;
  Vector& operator+=(const Vector&);
  Vector& operator*=(double);
  
  size_t length() const;
  
  // Console output via output stream
  friend std::ostream& operator<<(std::ostream&, const Vector&);

  // friend classes
  friend class Matrix;
  friend class DiagonalMatrix;

  // TODO: I do not want this. Extend Vector class so that this is not necessary
  friend class NeuralNetwork;
  
  // Matrix-vector operations
  friend Vector Matrix::operator*(const Vector&) const;
  Vector operator*(const Matrix&) const;
  Vector operator*(const DiagonalMatrix&) const;
  
  // free functions
  friend Rank1Matrix outer(const Vector&, const Vector&);
  friend DiagonalMatrix diag(const Vector&);
  friend double norm(const Vector&, double);

private:
  
};

// Class representing a diagonal matrix
class DiagonalMatrix {
public:
  DiagonalMatrix(const Vector&);

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
