#include "Image.h"

#include <iostream>

void Image::write(const std::string& filename, int width, int height, unsigned char** row_pointers)
{
  // Write image file
  FILE *fp = fopen(filename.c_str(), "wb");
  if(!fp)
    {
      std::cerr << "Error: Could not open image file\n";
      return;      
    }
  
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  
  png_init_io(png, fp);

  png_set_IHDR(png, info, width, height, 8,
               PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
               );

  png_write_info(png, info);
    
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  if (png && info)
    png_destroy_write_struct(&png, &info);

}


