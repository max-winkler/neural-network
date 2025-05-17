#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <png.h>
#include <string>
#include <vector>

#include "Matrix.h"

enum COLOR_TYPE {BW};

class Image
{
 public:

  static Image read(const std::string& filename);
  static Image from_matrix(const Matrix&);
  
  void write(const std::string& filename);

  Matrix to_matrix() const;
  
 private:

  Image() = delete;
  Image(int width, int height, std::vector<unsigned char>&& data);
  
  std::vector<unsigned char> data;
  size_t width;
  size_t height;
};

#endif
