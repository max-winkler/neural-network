#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <png.h>
#include <string>
#include <vector>

class Image
{
 public:
  static void write(const std::string& filename, int width, int height, unsigned char** row_pointers);
  static Image read(const std::string& filename);
 private:
  std::vector<unsigned char> pixels;
  size_t width;
  size_t height;
  int color_type;
  int bit_depth;
};

#endif
