#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>

#include "Matrix.h"

// Forward declarations
class DiagonalMatrix;

class Vector {
 public:
  // Constructor and destructor
  Vector();
  Vector(size_t);
  Vector(const Vector&);
  Vector(std::initializer_list<double>);
  ~Vector();

  Vector& operator=(const Vector&);
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
  
  // Console output via output stream
  friend std::ostream& operator<<(std::ostream&, const Vector&);

  // friend classes
  friend class DiagonalMatrix;
  
  // Matrix-vector operations
  friend Vector Matrix::operator*(const Vector&) const;
  Vector operator*(const Matrix&) const;
  Vector operator*(const DiagonalMatrix&) const;
  
  // free functions
  friend Matrix outer(const Vector&, const Vector&);

 private:
  
  double* data;
  size_t n;
};

class DiagonalMatrix {
public:
  DiagonalMatrix(const Vector&);

  friend class Vector;
private:
  size_t n;
  Vector diagonal;
};


#endif
