#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <png.h>
#include <string>

class Image
{
 public:
  static void write(const std::string& filename, int width, int height, unsigned char** row_pointers);
};

#endif
