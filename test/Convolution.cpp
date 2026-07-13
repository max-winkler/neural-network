#include <iostream>
#include <chrono>

#include "Matrix.h"
#include "LinAlg.h"
#include "Image.h"

class Timer {
public:
    void tic() {
        start = std::chrono::high_resolution_clock::now();
    }

    // Gibt vergangene Zeit in Millisekunden zurück und printet sie
    double toc(const std::string& label = "") {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << (label.empty() ? "Elapsed" : label) << ": " << ms << " ms\n";
        return ms;
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

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

  Timer t;
  t.tic();

  Matrix im_A = linalg::im2col(MatrixView(A), 3);
  Matrix im_K = linalg::im2col(MatrixView(K), 3);

  t.toc();
  t.tic();  
  A = linalg::matmul(im_A, im_K, true);
    
    //A = linalg::convolve_2(A, K, 1, 0, false);

  t.toc();  
  
  //A = linalg::pool(A, 2, POOLING_MAX);
  //A = linalg::convolve(A, K, 1, 0, false);
  //A = linalg::pool(A, 2, POOLING_MAX);

  
  int width_new = A.nCols();
  int height_new = A.nRows();
  
  std::cout << "A size before convolution: " << width << " x " << height << std::endl;
  std::cout << "A size after convolution : " << width_new << " x " << height_new << std::endl;

  Image output = Image::from_matrix(A);
  output.write("cat_bw.png");
       
  return 0;
}
