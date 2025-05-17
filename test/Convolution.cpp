#include <iostream>

#include "Matrix.h"
#include "LinAlg.h"
#include "Image.h"

int main()
{ 
  Image img = Image::read("cat.png");
  
  Matrix A = img.to_matrix();

  int width = A.nCols();
  int height = A.nRows();

  // manipulate image with filters
  Matrix K(3,3);
  K = {0, 1, 0,
    1, 4, 1,
    0, 1, 0};
  K *= (1./8);
  
  A = linalg::convolve(A, K, 1, 0, false);
  A = linalg::pool(A, 2, POOLING_MAX);
  A = linalg::convolve(A, K, 1, 0, false);
  A = linalg::pool(A, 2, POOLING_MAX);

  int width_new = A.nCols();
  int height_new = A.nRows();
  
  std::cout << "A size before convolution: " << width << " x " << height << std::endl;
  std::cout << "A size after convolution : " << width_new << " x " << height_new << std::endl;

  Image output = Image::from_matrix(A);
  output.write("cat_bw.png");
       
  return 0;
}
