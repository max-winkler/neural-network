#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>

#include "Matrix.h"

// Forward class declarations
class DiagonalMatrix;
class Rank1Matrix;
class Vector;
struct ScaledVector;

// Forward declaration of free functions
ScaledVector operator*(double, const Vector&);
double norm(const Vector&, double p=2.);
Rank1Matrix outer(const Vector&, const Vector&);

struct ScaledVector
{
  ScaledVector(double, const Vector&);
  
  double scale;
  const Vector* vector;
};

class Vector : public DataArray {
 public:
  // Constructors
  Vector();
  Vector(size_t);
  Vector(size_t, const double*);
  Vector(const Vector&);
  Vector(std::initializer_list<double>);

  // Destructor
  ~Vector();
  
  // Assignment operators
  Vector& operator=(const Vector&);
  Vector& operator=(Vector&&);
  Vector& operator=(std::initializer_list<double>);
  
  // Vector operations
  Vector operator+(const Vector&) const;
  Vector operator-(const Vector&) const;
  Vector& operator+=(const Vector&);
  Vector& operator+=(const ScaledVector&);
  Vector& operator*=(double);

  // Maximum and minimum functions
  size_t indMax() const;
  
  // Getters for basic properties
  size_t length() const;

  // Transform vector to matrix
  Matrix reshape(size_t, size_t) const;
  
  // Console output
  friend std::ostream& operator<<(std::ostream&, const Vector&);

  // Friend classes
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
