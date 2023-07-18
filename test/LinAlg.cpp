#include <iostream>

#include "Matrix.h"
#include "Vector.h"

int main()
{
  // Test vector class
  Vector b(5);

  b[0] = 1.;
  b[2] = 1.7490923435;
  b[3] = 4.1298234876;
  
  std::cout << "b = " << b << std::endl;

  // Test matrix class
  Matrix A(3, 5);

  A[0][2] = 3.734242198;
  A[0][4] = 3.213408721;
  A[1][1] = 1.982315208;
  A[2][4] = 1.092312100;
  std::cout << "A =\n" << A;

  Matrix B(2,2);
  B = {1.,2.,3.,4.};
  std::cout << "B =\n" << B;
  
  // Matrix vector operations
  Vector c = A*b;
  std::cout << "c = " << c << std::endl;

  Vector d = c*A;
  std::cout << "d = " << d << std::endl;

  // Vector vector operations
  Vector e(3);
  e[1] = 3.213897421;
  e[2] = 0.213098412;
  std::cout << "e = " << e << std::endl;
  
  Vector f = c + e;
  std::cout << "f = " << f << std::endl;

  Vector g(f);
  g += c;
  std::cout << "g = " << g << std::endl;
  return 0;
}
