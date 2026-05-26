#include "LinAlg.h"
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

  Matrix C(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) C(i, j) = 1.;
  std::cout << "C = \n" << C;
  
  // Linear algebra for matrices
  std::cout << "(B,C)_F = " << linalg::dot(B, C) << std::endl;
  std::cout << "B .* C =\n" << linalg::multiply(B, C) << std::endl;
  
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

  // Rank1Matrix
  Vector u({1.,1.});
  Vector v({1.,2.});
  
  Rank1Matrix D(u,v);
  B += D;
  std::cout << "B+(1 1)^T*(1 2) =\n" << B << std::endl;

  return 0;
}
