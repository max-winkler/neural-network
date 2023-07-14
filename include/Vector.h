#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>

#include "Matrix.h"

class Vector {
 public:
  // Constructor and destructor
  Vector(size_t);
  Vector(const Vector&);
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
  
  // Console output via output stream
  friend std::ostream& operator<<(std::ostream&, const Vector&);
  friend Vector Matrix::operator*(const Vector&) const;
 private:
  
  double* data;
  size_t n;
};

#endif
