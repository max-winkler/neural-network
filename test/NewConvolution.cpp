#include <iostream>

#include "LinAlg.h"

int main() {
  float data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  Tensor A(1, 4, 4, data);

  std::cout << "A:\n" << A << std::endl;

  float data_K[] = {1, 2, 3, 4};
  Tensor K(1,2,2, data);

  Matrix im_A = linalg::im2col(A, 2);
  std::cout << "Im(A):\n" << im_A << std::endl;
  
  Matrix im_K = linalg::im2col(K, 2);
  std::cout << "Im(K):\n" << im_K << std::endl;

  std::cout << "Old convolution:\n" << linalg::tensor_convolve(A, K);
  std::cout << "New convolution:\n" << linalg::matmul(im_A, im_K, true);
  
  return 0;  
}  
