#include <iostream>

#include "Matrix.h"

int main()
{
  Matrix A(6, 6);

  A = {
    1.0, 0.8, 0.6, 1.1, 0.6, 0.6,
    0.8, 0.7, 0.8, 0.8, 1.2, 0.5,
    1.3, 0.8, 0.6, 1.4, 0.6, 0.6,
    0.8, 0.7, 0.8, 0.8, 1.5, 0.5,
    0.7, 0.8, 0.6, 0.8, 0.6, 0.6,
    0.8, 1.6, 1.7, 0.8, 0.8, 1.8
  };

  Matrix B = A.pool();
  Matrix C = B.unpool(A);
  
  std::cout << "Original matrix:\n";
  std::cout << A;

  std::cout << "Pooled matrix:\n";
  std::cout << B;

  std::cout << "Unpooled matrix:\n";
  std::cout << C;

  return 0;
}
